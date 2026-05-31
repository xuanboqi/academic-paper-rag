from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import httpx
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

try:
    from embedding_model import DEFAULT_EMBEDDING_MODEL, load_embedding_model
except ModuleNotFoundError:
    from .embedding_model import DEFAULT_EMBEDDING_MODEL, load_embedding_model


DEFAULT_SYSTEM_PROMPT = (
    "You are a rigorous academic literature assistant. Answer in Chinese. "
    "Use only the provided paper passages. If the passages are insufficient, say so clearly. "
    "Cite sources with file names and page ranges."
)


@dataclass(frozen=True)
class SearchHit:
    """一次 pgvector 检索返回的证据片段。"""

    filename: str
    page_start: int
    page_end: int
    text: str
    score: float


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask a question over ingested literature with DeepSeek V4.")
    parser.add_argument("query")
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "8")))
    parser.add_argument("--deepseek-api-key", default=os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--deepseek-base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--deepseek-model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    return parser.parse_args()


def retrieve(dsn: str, query_embedding: list[float], top_k: int) -> list[SearchHit]:
    """用问题向量从 pgvector 召回 top-k 个最相关论文片段。"""
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    d.filename,
                    c.page_start,
                    c.page_end,
                    c.text,
                    1 - (c.embedding <=> %s::vector) AS score
                FROM literature_chunks c
                JOIN literature_documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()
    return [SearchHit(*row) for row in rows]


def build_context(hits: list[SearchHit]) -> str:
    """把召回片段整理成带编号的上下文，交给 DeepSeek 生成最终答案。"""
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        text = " ".join(hit.text.split())
        blocks.append(
            f"[{index}] filename={hit.filename}, pages={hit.page_start}-{hit.page_end}, "
            f"score={hit.score:.4f}\n{text}"
        )
    return "\n\n".join(blocks)


def ask_deepseek(
    api_key: str,
    base_url: str,
    model: str,
    query: str,
    hits: list[SearchHit],
) -> str:
    """调用 DeepSeek Chat Completions API。

    注意：这里不会把整个 Database 发给 DeepSeek，只发送 pgvector 召回的少量片段。
    """
    user_prompt = (
        f"问题：{query}\n\n"
        "参考文献片段：\n"
        f"{build_context(hits)}\n\n"
        "请给出结构化回答，并在关键结论后标注对应来源编号。"
    )
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=120) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"] or ""


def main() -> None:
    args = parse_args()
    if not args.postgres_dsn:
        raise SystemExit("POSTGRES_DSN is required. Copy .env.example to .env and edit it.")
    if not args.deepseek_api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required. Add it to .env first.")

    embedding_model = load_embedding_model(args.embedding_model)
    print(f"Embedding device: {embedding_model.device}")
    print(f"Embedding dimension: {embedding_model.dimension}")
    # RAG 在线问答链路：问题向量化 -> 向量库召回 -> DeepSeek 基于证据回答。
    query_embedding = embedding_model.encode_query(args.query)
    hits = retrieve(args.postgres_dsn, query_embedding, args.top_k)
    if not hits:
        raise SystemExit("No literature chunks found. Run scripts/ingest_literature.py first.")

    answer = ask_deepseek(
        api_key=args.deepseek_api_key,
        base_url=args.deepseek_base_url,
        model=args.deepseek_model,
        query=args.query,
        hits=hits,
    )
    print(answer)
    print("\nSources:")
    for index, hit in enumerate(hits, start=1):
        print(f"[{index}] {hit.filename} pp.{hit.page_start}-{hit.page_end} score={hit.score:.4f}")


if __name__ == "__main__":
    main()
