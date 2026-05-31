from __future__ import annotations

import argparse
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

try:
    from embedding_model import DEFAULT_EMBEDDING_MODEL, load_embedding_model
except ModuleNotFoundError:
    from .embedding_model import DEFAULT_EMBEDDING_MODEL, load_embedding_model


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Search literature chunks in PostgreSQL pgvector.")
    parser.add_argument("query")
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.postgres_dsn:
        raise SystemExit("POSTGRES_DSN is required. Copy .env.example to .env and edit it.")

    model = load_embedding_model(args.model)
    print(f"Embedding device: {model.device}")
    print(f"Embedding dimension: {model.dimension}")
    # 检索时只需要把用户问题向量化，不需要重新处理整篇论文库。
    query_embedding = model.encode_query(args.query)

    with psycopg.connect(args.postgres_dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # <=> 是 pgvector 的 cosine distance；距离越小越相似。
            # score = 1 - distance，方便人看，数值越大代表越相关。
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
                (query_embedding, query_embedding, args.top_k),
            )
            rows = cur.fetchall()

    for index, (filename, page_start, page_end, text, score) in enumerate(rows, start=1):
        preview = " ".join(text.split())[:600]
        print(f"\n[{index}] {filename} pp.{page_start}-{page_end} score={score:.4f}")
        print(preview)


if __name__ == "__main__":
    main()
