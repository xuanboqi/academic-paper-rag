from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from pypdf import PdfReader
from tqdm import tqdm

try:
    from embedding_model import DEFAULT_EMBEDDING_MODEL, LocalEmbeddingModel, load_embedding_model
except ModuleNotFoundError:
    from .embedding_model import DEFAULT_EMBEDDING_MODEL, LocalEmbeddingModel, load_embedding_model


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str


def sha256_file(path: Path) -> str:
    """用文件哈希作为文档稳定 ID，文件名改变后仍能识别同一篇 PDF。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    """清洗 PDF 抽取文本，减少换行、断词和控制字符对切片/embedding 的干扰。"""
    text = str(text)
    text = text.replace("\x00", " ")
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = re.sub(r"[\ud800-\udfff]", "", text)
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """逐页抽取 PDF 文本，并保留页码，方便问答结果回溯到论文页码。"""
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = clean_text(text)
        if text:
            pages.append((index, text))
    return pages


def split_words_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def chunk_pages(
    document_id: str,
    pages: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """把整篇论文按词数切成带重叠的 chunk。

    RAG 不直接把整篇论文塞给大模型，而是先把论文切成较小片段。
    overlap 用来保留跨边界语义，避免答案刚好被切断。
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text_parts: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for page_number, page_text in pages:
        if not page_text:
            continue
        start = cursor
        text_parts.append(page_text)
        cursor += len(page_text)
        page_ranges.append((start, cursor, page_number))
        text_parts.append("\n")
        cursor += 1

    full_text = "".join(text_parts).strip()
    words = split_words_with_offsets(full_text)
    if not words:
        return []

    chunks: list[Chunk] = []
    start_word = 0
    chunk_index = 0
    step = chunk_size - chunk_overlap

    while start_word < len(words):
        end_word = min(start_word + chunk_size, len(words))
        start_char = words[start_word][1]
        end_char = words[end_word - 1][2]
        chunk_text = clean_text(full_text[start_char:end_char])
        if not chunk_text:
            if end_word == len(words):
                break
            start_word += step
            continue
        touched_pages = [
            page
            for page_start, page_end, page in page_ranges
            if page_start < end_char and page_end > start_char
        ]
        chunks.append(
            Chunk(
                document_id=document_id,
                chunk_index=chunk_index,
                page_start=min(touched_pages) if touched_pages else 1,
                page_end=max(touched_pages) if touched_pages else 1,
                text=chunk_text,
            )
        )
        chunk_index += 1
        if end_word == len(words):
            break
        start_word += step

    return chunks


def batched(items: list[Chunk], batch_size: int) -> Iterable[list[Chunk]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def connect(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn)


def init_schema(conn: psycopg.Connection, vector_dim: int, recreate: bool = False) -> None:
    """初始化 PostgreSQL 表和 pgvector 索引。"""
    with conn.cursor() as cur:
        # vector 扩展提供 VECTOR 类型和 <=> 向量距离运算符。
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if recreate:
            cur.execute("DROP TABLE IF EXISTS literature_chunks")
            cur.execute("DROP TABLE IF EXISTS literature_documents")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS literature_documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                path TEXT NOT NULL,
                file_sha256 TEXT NOT NULL UNIQUE,
                page_count INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS literature_chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES literature_documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding VECTOR({vector_dim}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (document_id, chunk_index)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS literature_chunks_embedding_hnsw_idx
            ON literature_chunks
            USING hnsw (embedding vector_cosine_ops)
            """
        )
    conn.commit()


def upsert_document(
    conn: psycopg.Connection,
    document_id: str,
    path: Path,
    file_sha: str,
    page_count: int,
) -> None:
    """写入或更新文档元数据，真正的文本和向量保存在 literature_chunks。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO literature_documents (id, filename, path, file_sha256, page_count)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET filename = EXCLUDED.filename,
                path = EXCLUDED.path,
                file_sha256 = EXCLUDED.file_sha256,
                page_count = EXCLUDED.page_count
            """,
            (document_id, path.name, str(path.resolve()), file_sha, page_count),
        )


def document_has_chunks(conn: psycopg.Connection, document_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM literature_chunks WHERE document_id = %s)",
            (document_id,),
        )
        return bool(cur.fetchone()[0])


def replace_chunks(
    conn: psycopg.Connection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    """用当前切片结果替换某篇文档的旧 chunk，保证同一文档不会重复堆积。"""
    if not chunks:
        return
    document_id = chunks[0].document_id
    with conn.cursor() as cur:
        cur.execute("DELETE FROM literature_chunks WHERE document_id = %s", (document_id,))
        cur.executemany(
            """
            INSERT INTO literature_chunks
                (document_id, chunk_index, page_start, page_end, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.text,
                    embedding,
                )
                for chunk, embedding in zip(chunks, embeddings)
            ],
        )


def encode_chunk_batch(
    model: LocalEmbeddingModel,
    batch: list[Chunk],
    batch_size: int,
    pdf_name: str,
) -> tuple[list[Chunk], list[list[float]]]:
    """批量生成 chunk 向量；如果某个 batch 失败，自动降级为逐条重试。"""
    texts = [clean_text(chunk.text) for chunk in batch]
    try:
        return batch, model.encode_documents(texts, batch_size=batch_size)
    except Exception as exc:
        tqdm.write(f"Batch embedding failed for {pdf_name}: {exc}")
        tqdm.write("Retrying this batch one chunk at a time.")

    good_chunks: list[Chunk] = []
    embeddings: list[list[float]] = []
    for chunk, text in zip(batch, texts):
        try:
            vector = model.encode_documents([text], batch_size=1)[0]
        except Exception as exc:
            tqdm.write(
                f"Skipping bad chunk: {pdf_name} "
                f"chunk={chunk.chunk_index} pages={chunk.page_start}-{chunk.page_end} "
                f"chars={len(text)} error={exc}"
            )
            continue
        good_chunks.append(chunk)
        embeddings.append(vector)
    return good_chunks, embeddings


def ingest_pdf(
    conn: psycopg.Connection,
    model: LocalEmbeddingModel,
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    progress: Callable[[str, int, str], None] | None = None,
) -> tuple[int, int]:
    """单篇 PDF 的完整入库流程：哈希 -> 抽文本 -> 切片 -> 向量化 -> 写库。"""
    def report(stage: str, percent: int, message: str) -> None:
        if progress:
            progress(stage, percent, message)

    report("hash", 5, f"Hashing {pdf_path.name}")
    file_sha = sha256_file(pdf_path)
    document_id = file_sha[:32]
    report("extract_text", 15, f"Extracting text from {pdf_path.name}")
    tqdm.write(f"Extracting text: {pdf_path.name}")
    pages = extract_pages(pdf_path)
    report("chunk", 35, f"Chunking {len(pages)} extracted pages")
    chunks = chunk_pages(document_id, pages, chunk_size, chunk_overlap)
    tqdm.write(f"Embedding chunks: {pdf_path.name} ({len(chunks)} chunks)")

    upsert_document(conn, document_id, pdf_path, file_sha, len(pages))

    embeddings: list[list[float]] = []
    effective_batch_size = batch_size
    batches = list(batched(chunks, effective_batch_size))
    if not batches:
        report("embedding", 75, "No text chunks were generated")
    for batch_index, batch in enumerate(batches, start=1):
        percent = 35 + int((batch_index - 1) / max(len(batches), 1) * 45)
        report("embedding", percent, f"Embedding batch {batch_index}/{len(batches)} ({len(batch)} chunks)")
        tqdm.write(
            f"Embedding batch {batch_index}/{len(batches)} "
            f"({len(batch)} chunks): {pdf_path.name}"
        )
        good_batch, batch_embeddings = encode_chunk_batch(
            model=model,
            batch=batch,
            batch_size=effective_batch_size,
            pdf_name=pdf_path.name,
        )
        embeddings.extend(batch_embeddings)
        if len(good_batch) != len(batch):
            chunks = [
                chunk
                for chunk in chunks
                if chunk not in set(batch) or chunk in set(good_batch)
            ]

    report("write_db", 85, f"Writing {len(chunks)} chunks to PostgreSQL")
    tqdm.write(f"Writing chunks: {pdf_path.name}")
    replace_chunks(conn, chunks, embeddings)
    conn.commit()
    report("completed", 100, f"Completed {pdf_path.name}: {len(pages)} pages, {len(chunks)} chunks")
    return len(pages), len(chunks)


def ingest_pdf_if_needed(
    conn: psycopg.Connection,
    model: LocalEmbeddingModel,
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    skip_existing: bool,
) -> tuple[int, int, bool]:
    """支持断点续跑：已存在 chunk 的文档可以直接跳过。"""
    file_sha = sha256_file(pdf_path)
    document_id = file_sha[:32]
    if skip_existing and document_has_chunks(conn, document_id):
        tqdm.write(f"Skipping existing document: {pdf_path.name}")
        return 0, 0, True
    page_count, chunk_count = ingest_pdf(
        conn=conn,
        model=model,
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
    )
    return page_count, chunk_count, False


def find_pdfs(data_dir: Path, limit: int | None) -> list[Path]:
    pdfs = sorted(data_dir.rglob("*.pdf"))
    if limit is not None:
        return pdfs[:limit]
    return pdfs


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest academic PDFs into PostgreSQL pgvector.")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "Database"))
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "1200")))
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "200")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "32")))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.postgres_dsn:
        raise SystemExit("POSTGRES_DSN is required. Copy .env.example to .env and edit it.")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    pdfs = find_pdfs(data_dir, args.limit)
    if not pdfs:
        raise SystemExit(f"No PDF files found in {data_dir}")

    model = load_embedding_model(args.model)
    print(f"Embedding model: {args.model}")
    print(f"Embedding device: {model.device}")
    print(f"Embedding dimension: {model.dimension}")

    with connect(args.postgres_dsn) as conn:
        init_schema(conn, vector_dim=model.dimension, recreate=args.recreate)
        register_vector(conn)
        total_chunks = 0
        skipped = 0
        for pdf_path in tqdm(pdfs, desc="Ingesting PDFs"):
            page_count, chunk_count, was_skipped = ingest_pdf_if_needed(
                conn=conn,
                model=model,
                pdf_path=pdf_path,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size,
                skip_existing=args.skip_existing,
            )
            if was_skipped:
                skipped += 1
                continue
            total_chunks += chunk_count
            tqdm.write(f"{pdf_path.name}: {page_count} pages, {chunk_count} chunks")

    print(f"Done. Ingested {len(pdfs) - skipped} PDFs, skipped {skipped}, and wrote {total_chunks} chunks.")


if __name__ == "__main__":
    main()
