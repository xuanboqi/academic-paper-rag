from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

try:
    from embedding_model import DEFAULT_EMBEDDING_MODEL, load_embedding_model
except ModuleNotFoundError:
    from .embedding_model import DEFAULT_EMBEDDING_MODEL, load_embedding_model


@dataclass
class EvalCase:
    question: str
    expected_files: list[str]
    expected_terms: list[str]


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError(f"Line {line_number}: question is required")
            cases.append(
                EvalCase(
                    question=question,
                    expected_files=[str(item).lower() for item in payload.get("expected_files", [])],
                    expected_terms=[str(item).lower() for item in payload.get("expected_terms", [])],
                )
            )
    return cases


def search(conn: psycopg.Connection, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
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
        return list(cur.fetchall())


def evaluate_case(hits: list[dict[str, Any]], case: EvalCase) -> dict[str, Any]:
    filenames = [str(hit["filename"]).lower() for hit in hits]
    texts = [str(hit["text"]).lower() for hit in hits]
    expected_file_hit = (
        any(any(expected in filename for filename in filenames) for expected in case.expected_files)
        if case.expected_files
        else None
    )
    expected_term_hit = (
        any(any(term in text for text in texts) for term in case.expected_terms)
        if case.expected_terms
        else None
    )
    return {
        "question": case.question,
        "expected_file_hit": expected_file_hit,
        "expected_term_hit": expected_term_hit,
        "top_score": float(hits[0]["score"]) if hits else 0.0,
        "avg_score": float(mean(float(hit["score"]) for hit in hits)) if hits else 0.0,
        "hits": [
            {
                "filename": hit["filename"],
                "pages": f"{hit['page_start']}-{hit['page_end']}",
                "score": float(hit["score"]),
            }
            for hit in hits
        ],
    }


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate pgvector retrieval quality with JSONL test cases.")
    parser.add_argument("--cases", required=True, help="JSONL file with question, expected_files, expected_terms.")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "8")))
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.postgres_dsn:
        raise SystemExit("POSTGRES_DSN is required.")

    cases = load_cases(Path(args.cases))
    if not cases:
        raise SystemExit("No evaluation cases found.")

    model = load_embedding_model(args.model)
    results = []
    with psycopg.connect(args.postgres_dsn) as conn:
        register_vector(conn)
        for case in cases:
            query_embedding = model.encode_query(case.question)
            hits = search(conn, query_embedding, args.top_k)
            results.append(evaluate_case(hits, case))

    file_cases = [item for item in results if item["expected_file_hit"] is not None]
    term_cases = [item for item in results if item["expected_term_hit"] is not None]
    summary = {
        "case_count": len(results),
        "top_k": args.top_k,
        "file_hit_rate": (
            sum(1 for item in file_cases if item["expected_file_hit"]) / len(file_cases)
            if file_cases
            else None
        ),
        "term_hit_rate": (
            sum(1 for item in term_cases if item["expected_term_hit"]) / len(term_cases)
            if term_cases
            else None
        ),
        "avg_top_score": mean(item["top_score"] for item in results),
    }
    payload = {"summary": summary, "results": results}
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
