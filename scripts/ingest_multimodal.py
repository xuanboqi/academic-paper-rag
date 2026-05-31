from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.vision.local_qwen_vl import load_local_qwen_vl
from backend.vision.parser import render_pdf_pages
from scripts.embedding_model import load_embedding_model
from scripts.ingest_literature import init_schema, sha256_file, upsert_document


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Parse PDF figures/tables/formulas with local Qwen2.5-VL and store them in pgvector.")
    parser.add_argument("--pdf", type=Path, help="Process one PDF file. If omitted, process PDFs under DATA_DIR.")
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("DATA_DIR", "Database")))
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN"))
    parser.add_argument("--image-dir", type=Path, default=Path(os.getenv("MULTIMODAL_IMAGE_DIR", "artifacts/multimodal")))
    parser.add_argument("--dpi", type=int, default=int(os.getenv("MULTIMODAL_PAGE_DPI", "96")))
    parser.add_argument("--max-pages", type=int, default=int(os.getenv("MULTIMODAL_MAX_PAGES_PER_PDF", "10")))
    parser.add_argument("--max-candidate-pages", type=int, default=int(os.getenv("MULTIMODAL_MAX_CANDIDATE_PAGES", "3")))
    parser.add_argument("--all-pages", action="store_true", help="Render all pages up to --max-pages instead of only figure/table/equation candidates.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs when --pdf is not set.")
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "4")))
    parser.add_argument("--recreate", action="store_true", help="Delete existing multimodal chunks for each processed document first.")
    return parser.parse_args()


def init_multimodal_schema(conn: psycopg.Connection, vector_dim: int) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS literature_multimodal_chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES literature_documents(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                content_type TEXT NOT NULL DEFAULT 'page_vision',
                image_path TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding VECTOR({vector_dim}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (document_id, page_number, content_type)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS literature_multimodal_chunks_embedding_hnsw_idx
            ON literature_multimodal_chunks
            USING hnsw (embedding vector_cosine_ops)
            """
        )
    conn.commit()


def page_count(pdf_path: Path) -> int:
    import fitz

    with fitz.open(pdf_path) as doc:
        return len(doc)


def ensure_document(conn: psycopg.Connection, pdf_path: Path, file_sha: str, pages: int) -> str:
    """Return the existing document id for this PDF hash, or create a new metadata row."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM literature_documents WHERE file_sha256 = %s", (file_sha,))
        row = cur.fetchone()
    if row:
        return str(row[0])
    document_id = file_sha
    upsert_document(conn, document_id, pdf_path, file_sha, pages)
    conn.commit()
    return document_id


def process_pdf(
    conn: psycopg.Connection,
    pdf_path: Path,
    vision_model,
    embedding_model,
    image_root: Path,
    dpi: int,
    max_pages: int,
    max_candidate_pages: int,
    only_candidates: bool,
    batch_size: int,
    recreate: bool,
) -> int:
    file_sha = sha256_file(pdf_path)
    pages = page_count(pdf_path)
    document_id = ensure_document(conn, pdf_path, file_sha, pages)

    output_dir = image_root / document_id[:16]
    rendered_pages = render_pdf_pages(
        pdf_path=pdf_path,
        output_dir=output_dir,
        dpi=dpi,
        max_pages=max_pages,
        only_candidates=only_candidates,
        max_candidate_pages=max_candidate_pages,
    )
    if not rendered_pages:
        tqdm.write(f"{pdf_path.name}: no candidate pages found")
        return 0

    if recreate:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM literature_multimodal_chunks WHERE document_id = %s", (document_id,))
        conn.commit()

    inserted = 0
    for rendered in tqdm(rendered_pages, desc=f"Vision parsing: {pdf_path.name}", leave=False):
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM literature_multimodal_chunks
                    WHERE document_id = %s AND page_number = %s AND content_type = 'page_vision'
                )
                """,
                (document_id, rendered.page_number),
            )
            exists = bool(cur.fetchone()[0])
        if exists and not recreate:
            continue

        analysis = vision_model.analyze_page(rendered.image_path)
        if not analysis:
            continue
        text = (
            f"Multimodal page analysis for {pdf_path.name}, page {rendered.page_number}.\n"
            f"Candidate page: {rendered.is_candidate}. Candidate score: {rendered.score}.\n\n{analysis}"
        )
        embedding = embedding_model.encode_documents([text], batch_size=batch_size)[0]
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO literature_multimodal_chunks
                    (document_id, page_number, content_type, image_path, text, embedding)
                VALUES (%s, %s, 'page_vision', %s, %s, %s)
                ON CONFLICT (document_id, page_number, content_type) DO UPDATE
                SET image_path = EXCLUDED.image_path,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    created_at = now()
                """,
                (document_id, rendered.page_number, str(rendered.image_path.resolve()), text, embedding),
            )
        conn.commit()
        inserted += 1

    return inserted


def main() -> None:
    args = parse_args()
    if not args.postgres_dsn:
        raise SystemExit("POSTGRES_DSN is required.")

    pdfs = [args.pdf] if args.pdf else sorted((ROOT / args.data_dir).rglob("*.pdf"))
    if args.limit is not None:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit("No PDF files found.")

    embedding_model = load_embedding_model(os.getenv("EMBEDDING_MODEL"))
    vision_model = load_local_qwen_vl()
    with psycopg.connect(args.postgres_dsn) as conn:
        init_schema(conn, vector_dim=embedding_model.dimension, recreate=False)
        init_multimodal_schema(conn, vector_dim=embedding_model.dimension)
        register_vector(conn)
        total = 0
        for pdf_path in tqdm(pdfs, desc="Multimodal ingest"):
            pdf_path = pdf_path if pdf_path.is_absolute() else ROOT / pdf_path
            total += process_pdf(
                conn=conn,
                pdf_path=pdf_path,
                vision_model=vision_model,
                embedding_model=embedding_model,
                image_root=ROOT / args.image_dir,
                dpi=args.dpi,
                max_pages=args.max_pages,
                max_candidate_pages=args.max_candidate_pages,
                only_candidates=not args.all_pages and env_bool("MULTIMODAL_ONLY_CANDIDATE_PAGES", True),
                batch_size=args.batch_size,
                recreate=args.recreate,
            )
        print(f"Inserted/updated multimodal chunks: {total}")


if __name__ == "__main__":
    main()
