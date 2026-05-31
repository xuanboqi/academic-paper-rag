from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import psycopg
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pgvector.psycopg import register_vector
from pydantic import BaseModel

from backend.agent.graph import run_agentic_rag

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    # 复用 scripts/ 下已经写好的 embedding 加载逻辑，避免 Web API 和命令行脚本各写一套。
    sys.path.insert(0, str(SCRIPTS))

from embedding_model import load_embedding_model  # noqa: E402
from ingest_literature import ingest_pdf, init_schema, sha256_file  # noqa: E402
from rename_pdfs_by_title import extract_title, sanitize_filename, unique_target  # noqa: E402

load_dotenv(ROOT / ".env")

app = FastAPI(title="academic-paper-rag API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_embedding_model = None
_task_lock = threading.Lock()
# 向量化任务由 Web 页面触发后在后台子进程中运行；这些全局变量保存任务状态和日志。
_task_process: Optional[subprocess.Popen[str]] = None
_task_started_at: Optional[float] = None
_task_logs: deque[str] = deque(maxlen=600)
_task_status = "idle"
_ingest_jobs: dict[str, "IngestJob"] = {}
_ingest_lock = threading.Lock()


@dataclass
class IngestJob:
    id: str
    kind: str
    filename: str
    path: str
    status: str = "queued"
    stage: str = "queued"
    percent: int = 0
    message: str = "Queued"
    duplicate: bool = False
    existing_filename: Optional[str] = None
    warning: Optional[str] = None
    error: Optional[str] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8


class AskRequest(BaseModel):
    query: str
    top_k: int = 8
    deepseek_model: Optional[str] = None


class AgentAskResponse(BaseModel):
    answer: str
    hits: list[dict[str, Any]]
    model: str
    task_type: str
    rewritten_query: str
    steps: list[str]


class VectorizeRequest(BaseModel):
    quick_test: bool = False
    recreate: bool = False
    skip_existing: bool = True
    batch_size: int = 4
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    limit: Optional[int] = None


class UploadIngestResponse(BaseModel):
    original_filename: str
    filename: str
    path: str
    title: Optional[str]
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    task_id: Optional[str] = None
    status: str = "queued"
    duplicate: bool = False
    existing_filename: Optional[str] = None
    warning: Optional[str] = None


class RevectorizeRequest(BaseModel):
    path: str


class RevectorizeResponse(BaseModel):
    filename: str
    path: str
    task_id: str
    status: str


class IngestTaskResponse(BaseModel):
    id: str
    kind: str
    filename: str
    path: str
    status: str
    stage: str
    percent: int
    message: str
    duplicate: bool = False
    existing_filename: Optional[str] = None
    warning: Optional[str] = None
    error: Optional[str] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    started_at: float
    finished_at: Optional[float] = None


def dsn() -> str:
    value = os.getenv("POSTGRES_DSN")
    if not value:
        raise HTTPException(status_code=500, detail="POSTGRES_DSN is not configured")
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def data_dir() -> Path:
    return ROOT / os.getenv("DATA_DIR", "Database")


def unique_upload_path(target_dir: Path, filename: str) -> Path:
    """为上传文件找一个不覆盖旧文件的临时保存路径。"""
    original = target_dir / Path(filename).name
    stem = original.stem or "uploaded"
    suffix = original.suffix or ".pdf"
    candidate = original
    counter = 2
    while candidate.exists():
        candidate = target_dir / f"{stem} ({counter}){suffix}"
        counter += 1
    return candidate


def connect() -> psycopg.Connection:
    return psycopg.connect(dsn(), connect_timeout=5)


def validate_pdf_path(raw_path: str) -> Path:
    pdf_path = Path(raw_path).expanduser().resolve()
    root = data_dir().resolve()
    try:
        pdf_path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="PDF path must be inside the data directory") from exc
    if not pdf_path.exists() or not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="PDF file not found")
    if pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    return pdf_path


def get_embedding_model():
    """懒加载 embedding 模型，避免 FastAPI 启动时就占用显存。"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = load_embedding_model(os.getenv("EMBEDDING_MODEL"))
    return _embedding_model


def find_document_by_sha(file_sha: str) -> Optional[dict[str, Any]]:
    rows = fetch_all(
        """
        SELECT
            d.id,
            d.filename,
            d.path,
            d.page_count,
            count(c.id)::int AS chunk_count
        FROM literature_documents d
        LEFT JOIN literature_chunks c ON c.document_id = d.id
        WHERE d.file_sha256 = %s
        GROUP BY d.id
        """,
        (file_sha,),
    )
    return rows[0] if rows else None


def job_snapshot(job: IngestJob) -> IngestTaskResponse:
    return IngestTaskResponse(**job.__dict__)


def update_ingest_job(job_id: str, **changes: Any) -> None:
    with _ingest_lock:
        job = _ingest_jobs[job_id]
        for key, value in changes.items():
            setattr(job, key, value)


def ingest_existing_pdf(pdf_path: Path, job_id: Optional[str] = None) -> tuple[int, int]:
    """Run the same PDF -> chunks -> embeddings -> pgvector pipeline used by uploads."""
    model = get_embedding_model()
    chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    batch_size = int(os.getenv("BATCH_SIZE", "4"))

    def progress(stage: str, percent: int, message: str) -> None:
        if job_id:
            update_ingest_job(job_id, stage=stage, percent=percent, message=message)

    with connect() as conn:
        init_schema(conn, vector_dim=model.dimension, recreate=False)
        register_vector(conn)
        return ingest_pdf(
            conn=conn,
            model=model,
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            progress=progress,
        )


def run_ingest_job(job_id: str, pdf_path: Path) -> None:
    update_ingest_job(job_id, status="running", stage="starting", percent=1, message="Starting ingest")
    try:
        page_count, chunk_count = ingest_existing_pdf(pdf_path, job_id=job_id)
        warning = None
        if page_count == 0 or chunk_count == 0:
            warning = "No text chunks were extracted. This may be a scanned PDF and may need OCR."
        update_ingest_job(
            job_id,
            status="completed",
            stage="completed",
            percent=100,
            message=f"Completed: {page_count} pages / {chunk_count} chunks",
            page_count=page_count,
            chunk_count=chunk_count,
            warning=warning,
            finished_at=time.time(),
        )
    except Exception as exc:
        update_ingest_job(
            job_id,
            status="failed",
            stage="failed",
            percent=100,
            message="Ingest failed",
            error=str(exc),
            finished_at=time.time(),
        )


def start_ingest_job(kind: str, pdf_path: Path, duplicate: bool = False, existing_filename: Optional[str] = None) -> IngestJob:
    job_id = uuid.uuid4().hex
    job = IngestJob(
        id=job_id,
        kind=kind,
        filename=pdf_path.name,
        path=str(pdf_path.resolve()),
        duplicate=duplicate,
        existing_filename=existing_filename,
    )
    with _ingest_lock:
        _ingest_jobs[job_id] = job
    thread = threading.Thread(target=run_ingest_job, args=(job_id, pdf_path), daemon=True)
    thread.start()
    return job


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with connect() as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(query, params)
            return list(cur.fetchall())


def ingest_job_running() -> bool:
    with _ingest_lock:
        return any(job.status in {"queued", "running"} for job in _ingest_jobs.values())


def db_counts() -> dict[str, Any]:
    """读取首页状态栏需要的文档数和 chunk 数。"""
    try:
        rows = fetch_all(
            """
            SELECT
                (SELECT count(*) FROM literature_documents) AS documents,
                (SELECT count(*) FROM literature_chunks) AS chunks,
                (SELECT count(*) FROM literature_multimodal_chunks) AS multimodal_chunks
            """
        )
    except Exception:
        try:
            rows = fetch_all(
                """
                SELECT
                    (SELECT count(*) FROM literature_documents) AS documents,
                    (SELECT count(*) FROM literature_chunks) AS chunks
                """
            )
            return {
                "documents": int(rows[0]["documents"]),
                "chunks": int(rows[0]["chunks"]),
                "multimodal_chunks": 0,
                "connected": True,
            }
        except Exception:
            return {"documents": 0, "chunks": 0, "multimodal_chunks": 0, "connected": False}
    return {
        "documents": int(rows[0]["documents"]),
        "chunks": int(rows[0]["chunks"]),
        "multimodal_chunks": int(rows[0]["multimodal_chunks"]),
        "connected": True,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    counts = db_counts()
    return {
        "database": "connected" if counts["connected"] else "unavailable",
        "documents": counts["documents"],
        "chunks": counts["chunks"],
        "multimodal_chunks": counts["multimodal_chunks"],
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "embedding_device": os.getenv("EMBEDDING_DEVICE", "auto"),
        "embedding_dimension": 1024,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deepseek_model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    }


@app.get("/api/library/documents")
def documents() -> dict[str, Any]:
    """合并文件系统中的 PDF 与数据库中的入库记录，供“文献管理”页展示。"""
    pdfs = {path.name: path for path in sorted(data_dir().rglob("*.pdf"))}
    ingested: dict[str, dict[str, Any]] = {}
    ingested_by_path: dict[str, dict[str, Any]] = {}
    ingested_by_sha: dict[str, dict[str, Any]] = {}
    try:
        rows = fetch_all(
            """
            SELECT
                d.id,
                d.filename,
                d.path,
                d.file_sha256,
                d.page_count,
                d.created_at,
                count(DISTINCT c.id)::int AS chunk_count,
                count(DISTINCT m.id)::int AS multimodal_chunk_count
            FROM literature_documents d
            LEFT JOIN literature_chunks c ON c.document_id = d.id
            LEFT JOIN literature_multimodal_chunks m ON m.document_id = d.id
            GROUP BY d.id
            ORDER BY d.filename
            """
        )
        for row in rows:
            ingested[row["filename"]] = row
            ingested_by_path[str(Path(row["path"]).resolve())] = row
            ingested_by_sha[row["file_sha256"]] = row
    except Exception:
        rows = []

    merged = []
    for filename, path in pdfs.items():
        resolved_path = str(path.resolve())
        row = ingested.get(filename) or ingested_by_path.get(resolved_path)
        if row is None:
            try:
                file_sha = sha256_file(path)
            except Exception:
                file_sha = None
            row = ingested_by_sha.get(file_sha) if file_sha else None
        merged.append(
            {
                "id": row["id"] if row else None,
                "filename": filename,
                "path": str(path),
                "page_count": row["page_count"] if row else None,
                "chunk_count": row["chunk_count"] if row else 0,
                "multimodal_chunk_count": row["multimodal_chunk_count"] if row and "multimodal_chunk_count" in row else 0,
                "created_at": str(row["created_at"]) if row else None,
                "status": "indexed" if row and row["chunk_count"] else "pending",
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }
        )

    pdf_paths = {str(path.resolve()) for path in pdfs.values()}
    for filename, row in ingested.items():
        if filename not in pdfs and str(Path(row["path"]).resolve()) not in pdf_paths:
            merged.append(
                {
                    "id": row["id"],
                    "filename": filename,
                    "path": row["path"],
                    "page_count": row["page_count"],
                    "chunk_count": row["chunk_count"],
                    "multimodal_chunk_count": row["multimodal_chunk_count"] if "multimodal_chunk_count" in row else 0,
                    "created_at": str(row["created_at"]),
                    "status": "missing-file",
                    "size_mb": None,
                }
            )
    return {"documents": merged, "pdf_count": len(pdfs)}


@app.get("/api/library/documents/{document_id}/chunks")
def document_chunks(document_id: str) -> dict[str, Any]:
    rows = fetch_all(
        """
        SELECT chunk_index, page_start, page_end, text
        FROM literature_chunks
        WHERE document_id = %s
        ORDER BY chunk_index
        """,
        (document_id,),
    )
    return {"chunks": rows}


@app.post("/api/documents/upload")
def upload_document(file: UploadFile) -> UploadIngestResponse:
    """上传 PDF，按论文标题重命名，并立即切片、向量化、写入 pgvector。"""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if _task_process is not None or ingest_job_running():
        raise HTTPException(status_code=409, detail="Vectorization task is running. Try again later.")

    target_dir = data_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    original_filename = Path(file.filename).name
    temporary_target = unique_upload_path(target_dir, original_filename)
    with temporary_target.open("wb") as handle:
        handle.write(file.file.read())

    title = extract_title(temporary_target, max_pages=2)
    final_target = temporary_target
    if title:
        final_target = unique_target(
            temporary_target,
            sanitize_filename(title, max_length=180),
            planned_targets=set(),
        )
        if final_target != temporary_target:
            temporary_target.rename(final_target)

    file_sha = sha256_file(final_target)
    existing = find_document_by_sha(file_sha)
    duplicate = bool(existing and int(existing["chunk_count"]) > 0)
    warning = None
    if duplicate:
        warning = f"Duplicate PDF detected. Reusing existing vectors from {existing['filename']}."
        page_count = int(existing["page_count"])
        chunk_count = int(existing["chunk_count"])
        status = "completed"
        task_id = None
    else:
        job = start_ingest_job("upload", final_target)
        page_count = None
        chunk_count = None
        status = job.status
        task_id = job.id

    return UploadIngestResponse(
        original_filename=original_filename,
        filename=final_target.name,
        path=str(final_target.resolve()),
        title=title,
        page_count=page_count,
        chunk_count=chunk_count,
        task_id=task_id,
        status=status,
        duplicate=duplicate,
        existing_filename=existing["filename"] if existing else None,
        warning=warning,
    )


@app.post("/api/library/revectorize")
def revectorize_document(request: RevectorizeRequest) -> RevectorizeResponse:
    """Rebuild chunks and embeddings for one existing PDF in Database/."""
    if _task_process is not None or ingest_job_running():
        raise HTTPException(status_code=409, detail="Vectorization task is running. Try again later.")
    pdf_path = validate_pdf_path(request.path)
    job = start_ingest_job("revectorize", pdf_path)
    return RevectorizeResponse(
        filename=pdf_path.name,
        path=str(pdf_path),
        task_id=job.id,
        status=job.status,
    )


@app.get("/api/ingest/tasks/{task_id}")
def ingest_task_status(task_id: str) -> IngestTaskResponse:
    with _ingest_lock:
        job = _ingest_jobs.get(task_id)
        if not job:
            raise HTTPException(status_code=404, detail="Ingest task not found")
        return job_snapshot(job)


def retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
    """Web 端复用的检索函数：问题 -> Qwen3 向量 -> pgvector top-k。

    如果已经运行过多模态入库脚本，会同时检索正文 chunks 和图表/公式视觉解析 chunks。
    """
    model = get_embedding_model()
    query_embedding = model.encode_query(query)
    hits: list[dict[str, Any]] = []
    with connect() as conn:
        register_vector(conn)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT
                    d.filename,
                    c.page_start,
                    c.page_end,
                    c.text,
                    1 - (c.embedding <=> %s::vector) AS score,
                    'text' AS source_type
                FROM literature_chunks c
                JOIN literature_documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            hits.extend(list(cur.fetchall()))
            if env_bool("INCLUDE_MULTIMODAL_RETRIEVAL", True):
                try:
                    cur.execute(
                        """
                        SELECT
                            d.filename,
                            m.page_number AS page_start,
                            m.page_number AS page_end,
                            m.text,
                            1 - (m.embedding <=> %s::vector) AS score,
                            'multimodal' AS source_type
                        FROM literature_multimodal_chunks m
                        JOIN literature_documents d ON d.id = m.document_id
                        ORDER BY m.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embedding, query_embedding, top_k),
                    )
                    hits.extend(list(cur.fetchall()))
                except psycopg.errors.UndefinedTable:
                    conn.rollback()
    hits.sort(key=lambda item: float(item["score"]), reverse=True)
    return hits[:top_k]


@app.post("/api/retrieval/search")
def search(request: SearchRequest) -> dict[str, Any]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    return {"hits": retrieve(request.query, request.top_k)}


def build_context(hits: list[dict[str, Any]]) -> str:
    """把检索结果组织为 DeepSeek 可读的证据上下文。"""
    blocks = []
    for index, hit in enumerate(hits, start=1):
        text = " ".join(hit["text"].split())
        blocks.append(
            f"[{index}] type={hit.get('source_type', 'text')}, filename={hit['filename']}, pages={hit['page_start']}-{hit['page_end']}, "
            f"score={float(hit['score']):.4f}\n{text}"
        )
    return "\n\n".join(blocks)


def chat_deepseek(messages: list[dict[str, str]], model: Optional[str] = None, temperature: float = 0.2) -> str:
    """Call DeepSeek Chat Completions with a prepared message list."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is not configured")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    selected_model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    payload = {
        "model": selected_model,
        "temperature": temperature,
        "messages": messages,
    }
    with httpx.Client(timeout=120) as client:
        response = client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"] or ""


@app.post("/api/qa/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    """完整问答接口：先检索证据，再让 DeepSeek 基于证据生成答案。"""
    hits = retrieve(request.query, request.top_k)
    model = request.deepseek_model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    answer = chat_deepseek(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是严谨的学术文献助手。只能基于给定文献片段回答。"
                    "如果片段不足以回答，要明确说明信息不足。回答时用中文，并引用来源编号。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{request.query}\n\n"
                    f"参考文献片段：\n{build_context(hits)}\n\n"
                    "请给出结构化回答，并在关键结论后标注对应来源编号。"
                ),
            },
        ],
    )
    return {"answer": answer, "hits": hits, "model": model}


@app.post("/api/agent/ask", response_model=AgentAskResponse)
def agent_ask(request: AskRequest) -> AgentAskResponse:
    """Agentic RAG: classify task -> rewrite query -> retrieve evidence -> generate answer."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    model = request.deepseek_model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    def chat(messages: list[dict[str, str]], temperature: float) -> str:
        return chat_deepseek(messages, model=model, temperature=temperature)

    try:
        state = run_agentic_rag(
            query=request.query,
            top_k=request.top_k,
            chat=chat,
            retrieve=retrieve,
            build_context=build_context,
            model=model,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AgentAskResponse(
        answer=state.get("answer", ""),
        hits=state.get("hits", []),
        model=model,
        task_type=state.get("task_type", "qa"),
        rewritten_query=state.get("rewritten_query", request.query),
        steps=state.get("steps", []),
    )


def run_vectorize(command: list[str]) -> None:
    """后台执行 train_literature_rag.py，并把 stdout 持续写入内存日志。"""
    global _task_process, _task_status
    try:
        _task_logs.append("$ " + " ".join(command))
        _task_process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert _task_process.stdout is not None
        for line in _task_process.stdout:
            _task_logs.append(line.rstrip())
        code = _task_process.wait()
        _task_status = "completed" if code == 0 else f"failed:{code}"
    except Exception as exc:
        _task_logs.append(f"Task error: {exc}")
        _task_status = "failed"
    finally:
        _task_process = None


@app.post("/api/vectorize/start")
def start_vectorize(request: VectorizeRequest) -> dict[str, Any]:
    """从 Web 页面启动向量化任务，参数会转换成命令行脚本参数。"""
    global _task_process, _task_started_at, _task_status
    with _task_lock:
        if _task_process is not None:
            raise HTTPException(status_code=409, detail="Vectorization is already running")
        command = [sys.executable, str(SCRIPTS / "train_literature_rag.py"), "--batch-size", str(request.batch_size)]
        if request.quick_test:
            command.append("--quick-test")
        if request.recreate:
            command.append("--recreate")
        if not request.skip_existing:
            command.append("--no-skip-existing")
        if request.limit is not None:
            command.extend(["--limit", str(request.limit)])
        if request.chunk_size is not None:
            command.extend(["--chunk-size", str(request.chunk_size)])
        if request.chunk_overlap is not None:
            command.extend(["--chunk-overlap", str(request.chunk_overlap)])
        _task_logs.clear()
        _task_status = "running"
        _task_started_at = time.time()
        thread = threading.Thread(target=run_vectorize, args=(command,), daemon=True)
        thread.start()
    return {"status": _task_status, "command": command}


@app.post("/api/vectorize/stop")
def stop_vectorize() -> dict[str, str]:
    global _task_process, _task_status
    if _task_process is not None:
        _task_process.terminate()
        _task_status = "stopping"
    return {"status": _task_status}


@app.get("/api/vectorize/status")
def vectorize_status() -> dict[str, Any]:
    return {
        "status": _task_status,
        "started_at": _task_started_at,
        "running": _task_process is not None,
        "logs": list(_task_logs)[-200:],
    }
