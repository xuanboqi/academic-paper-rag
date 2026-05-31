from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import psycopg
import torch
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from tqdm import tqdm

try:
    from embedding_model import DEFAULT_EMBEDDING_MODEL, load_embedding_model
    from ingest_literature import (
        connect,
        find_pdfs,
        ingest_pdf_if_needed,
        init_schema,
    )
except ModuleNotFoundError:
    from .embedding_model import DEFAULT_EMBEDDING_MODEL, load_embedding_model
    from .ingest_literature import (
        connect,
        find_pdfs,
        ingest_pdf_if_needed,
        init_schema,
    )


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Build the academic-paper-rag vector database. This does not fine-tune Qwen3; "
            "it embeds the PDF corpus and stores vectors in PostgreSQL."
        )
    )
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "Database"))
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "1200")))
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "200")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "32")))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--require-cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Shortcut for --limit 3 --recreate --batch-size 4.",
    )
    return parser.parse_args()


def apply_presets(args: argparse.Namespace) -> argparse.Namespace:
    """常用小样本测试参数，等价于 --limit 3 --recreate --batch-size 4。"""
    if args.quick_test:
        args.limit = 3
        args.recreate = True
        args.batch_size = 4
    return args


def validate_runtime(args: argparse.Namespace) -> None:
    """打印并检查运行环境，避免误用没有 CUDA 的 Python 环境慢速入库。"""
    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif args.require_cuda:
        raise SystemExit(
            "CUDA is not available in this Python environment. "
            "Activate your CUDA-enabled Conda environment or pass --allow-cpu."
        )


def main() -> None:
    args = apply_presets(parse_args())
    if not args.postgres_dsn:
        raise SystemExit("POSTGRES_DSN is required. Copy .env.example to .env and edit it.")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    validate_runtime(args)
    pdfs = find_pdfs(data_dir, args.limit)
    if not pdfs:
        raise SystemExit(f"No PDF files found in {data_dir}")

    model = load_embedding_model(args.model)
    print(f"Embedding model: {args.model}")
    print(f"Embedding device: {model.device}")
    print(f"Embedding dimension: {model.dimension}")
    print(f"PDF count: {len(pdfs)}")
    print(f"Chunk size / overlap: {args.chunk_size} / {args.chunk_overlap}")
    print(f"Batch size: {args.batch_size}")
    print(f"Skip existing: {args.skip_existing}")

    started = time.perf_counter()
    total_chunks = 0
    completed = 0
    skipped = 0

    with connect(args.postgres_dsn) as conn:
        # 建表时必须使用当前 embedding 维度；换模型导致维度变化时要 --recreate。
        init_schema(conn, vector_dim=model.dimension, recreate=args.recreate)
        register_vector(conn)
        for pdf_path in tqdm(pdfs, desc="Building vector DB"):
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
            completed += 1
            total_chunks += chunk_count
            tqdm.write(f"Completed: {pdf_path.name} ({page_count} pages, {chunk_count} chunks)")

    elapsed = time.perf_counter() - started
    print(
        f"Done. Completed {completed} PDFs, skipped {skipped}, "
        f"wrote {total_chunks} chunks in {elapsed / 60:.1f} minutes."
    )


if __name__ == "__main__":
    main()
