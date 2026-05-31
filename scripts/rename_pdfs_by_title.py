from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import psycopg
from dotenv import load_dotenv
from pypdf import PdfReader

INVALID_FILENAME_CHARS = r'<>:"/\|?*'
GENERIC_TITLE_PATTERNS = (
    "sciencedirect",
    "elsevier",
    "contents lists available",
    "journal homepage",
    "provided proper attribution",
    "grants permission",
    "reproduce the tables",
    "microsoft word",
    "untitled",
    "main",
)
STOP_MARKERS = (
    "abstract",
    "keywords",
    "index terms",
    "introduction",
    "1 introduction",
    "received",
    "available online",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_line(line: str) -> str:
    line = line.replace("\x00", " ")
    line = normalize_spaces(line)
    line = re.sub(r"^\d+\s*$", "", line)
    return line.strip(" \t\r\n")


def is_bad_title(text: str) -> bool:
    value = normalize_spaces(text).lower()
    if len(value) < 12 or len(value) > 260:
        return True
    if not re.search(r"[A-Za-z]", value):
        return True
    if any(pattern in value for pattern in GENERIC_TITLE_PATTERNS):
        return True
    if value.startswith(("http", "doi", "www.")):
        return True
    return False


def metadata_title(reader: PdfReader) -> Optional[str]:
    """优先读取 PDF 元数据中的标题，这是最可靠、最少误判的来源。"""
    try:
        title = reader.metadata.title if reader.metadata else None
    except Exception:
        title = None
    if title and not is_bad_title(title):
        return normalize_spaces(title)
    return None


def iter_first_page_lines(reader: PdfReader, max_pages: int) -> Iterable[str]:
    for page in reader.pages[:max_pages]:
        text = page.extract_text() or ""
        for raw_line in text.splitlines():
            line = clean_line(raw_line)
            if line:
                yield line


def looks_like_author_or_affiliation(line: str) -> bool:
    lower = line.lower()
    if "@" in line:
        return True
    if any(word in lower for word in ("university", "institute", "department", "school of", "laboratory")):
        return True
    if re.search(r"\b(and|,)\b", lower) and len(line.split()) <= 14:
        return True
    if re.search(r"\d", line) and len(line.split()) <= 8:
        return True
    return False


def looks_like_following_author_line(line: str) -> bool:
    if any(marker in line for marker in ("∗", "†", "*")):
        return True
    words = line.split()
    if len(words) <= 6 and re.search(r"\b[A-Z][a-z]+\.?\b\s+\b[A-Z][A-Za-z.-]+\b", line):
        title_words = {"the", "and", "for", "with", "via", "from", "under", "using", "based", "towards"}
        if not any(word.lower().strip(":-") in title_words for word in words):
            return True
    return False


def first_page_title(reader: PdfReader, max_pages: int) -> Optional[str]:
    """从首页文本推断标题，处理没有 metadata title 的论文。"""
    lines = list(iter_first_page_lines(reader, max_pages=max_pages))[:80]
    candidates: list[str] = []

    for index, line in enumerate(lines):
        lower = line.lower()
        if any(marker == lower or lower.startswith(marker + " ") for marker in STOP_MARKERS):
            break
        if is_bad_title(line):
            continue
        if looks_like_author_or_affiliation(line):
            continue

        block = [line]
        for next_line in lines[index + 1 : index + 5]:
            next_lower = next_line.lower()
            if any(marker == next_lower or next_lower.startswith(marker + " ") for marker in STOP_MARKERS):
                break
            if looks_like_author_or_affiliation(next_line) or looks_like_following_author_line(next_line):
                break
            if is_bad_title(next_line):
                break
            if len(" ".join(block + [next_line])) > 240:
                break
            block.append(next_line)

        candidate = normalize_spaces(" ".join(block))
        if not is_bad_title(candidate):
            candidates.append(candidate)

    if not candidates:
        return None

    candidates.sort(key=lambda item: (len(item.split()), len(item)), reverse=True)
    return candidates[0]


def extract_title(path: Path, max_pages: int) -> Optional[str]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        print(f"[SKIP] Cannot read PDF: {path.name} ({exc})")
        return None

    return metadata_title(reader) or first_page_title(reader, max_pages=max_pages)


def sanitize_filename(title: str, max_length: int) -> str:
    """把论文标题转换为 Windows 可用的 PDF 文件名。"""
    table = str.maketrans({char: " " for char in INVALID_FILENAME_CHARS})
    name = title.translate(table)
    name = re.sub(r"[\x00-\x1f]", " ", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" .")
    if len(name) > max_length:
        name = name[:max_length].rstrip(" .")
    return name or "untitled"


def unique_target(path: Path, desired_stem: str, planned_targets: set[Path]) -> Path:
    target = path.with_name(f"{desired_stem}.pdf")
    counter = 2
    while target.exists() or target in planned_targets:
        if target.resolve() == path.resolve():
            return target
        target = path.with_name(f"{desired_stem} ({counter}).pdf")
        counter += 1
    return target


def update_database(dsn: str, old_path: Path, new_path: Path) -> None:
    """文件改名后同步更新数据库中的 filename/path，前端显示才会一致。"""
    file_sha = sha256_file(new_path)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE literature_documents
                SET filename = %s,
                    path = %s
                WHERE file_sha256 = %s
                """,
                (new_path.name, str(new_path.resolve()), file_sha),
            )
            updated = cur.rowcount
        conn.commit()
    if updated:
        print(f"[DB] Updated literature_documents for {new_path.name}")
    else:
        print(f"[DB] No matching row found for {old_path.name}")


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Rename PDFs using paper titles.")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "Database"))
    parser.add_argument("--apply", action="store_true", help="Actually rename files. Default is dry-run.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=180)
    parser.add_argument(
        "--no-db-update",
        action="store_true",
        help="Do not update literature_documents after renaming.",
    )
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    pdfs = sorted(data_dir.rglob("*.pdf"))
    if args.limit is not None:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit(f"No PDF files found in {data_dir}")

    planned_targets: set[Path] = set()
    operations: list[tuple[Path, Path, str]] = []

    for path in pdfs:
        title = extract_title(path, max_pages=args.max_pages)
        if not title:
            print(f"[SKIP] Title not found: {path.name}")
            continue
        target = unique_target(path, sanitize_filename(title, args.max_length), planned_targets)
        planned_targets.add(target)
        if target.name == path.name:
            print(f"[OK]   Already named: {path.name}")
            continue
        operations.append((path, target, title))
        print(f"[PLAN] {path.name}")
        print(f"       -> {target.name}")

    print("")
    print(f"Planned renames: {len(operations)}")
    if not args.apply:
        print("Dry-run only. Add --apply to rename files.")
        return

    for old_path, new_path, _title in operations:
        old_path.rename(new_path)
        print(f"[DONE] {old_path.name} -> {new_path.name}")
        if args.postgres_dsn and not args.no_db_update:
            update_database(args.postgres_dsn, old_path, new_path)


if __name__ == "__main__":
    main()
