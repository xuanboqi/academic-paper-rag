from __future__ import annotations

import argparse
import ctypes
import gc
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

try:
    import resource
except ModuleNotFoundError:
    resource = None

try:
    from embedding_model import DEFAULT_EMBEDDING_MODEL, LocalEmbeddingModel, load_embedding_model
    from ingest_literature import (
        batched,
        chunk_pages,
        connect,
        extract_pages,
        init_schema,
        replace_chunks,
        sha256_file,
        upsert_document,
    )
except ModuleNotFoundError:
    from .embedding_model import DEFAULT_EMBEDDING_MODEL, LocalEmbeddingModel, load_embedding_model
    from .ingest_literature import (
        batched,
        chunk_pages,
        connect,
        extract_pages,
        init_schema,
        replace_chunks,
        sha256_file,
        upsert_document,
    )


@dataclass
class StageStats:
    seconds: float = 0.0
    rss_delta_mb: float = 0.0
    gpu_alloc_delta_mb: float = 0.0


@dataclass
class PdfProfile:
    filename: str
    size_mb: float
    pages: int = 0
    chunks: int = 0
    stages: dict[str, StageStats] = field(default_factory=dict)
    total_seconds: float = 0.0
    peak_rss_mb: float = 0.0
    peak_gpu_alloc_mb: float = 0.0


def process_rss_mb() -> float:
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    if os.name == "nt":
        return windows_rss_mb()
    if resource is None:
        return 0.0
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except Exception:
        return 0.0
    # Windows/POSIX 的 ru_maxrss 单位不完全一致；这里作为无 psutil 时的近似值。
    value = float(usage.ru_maxrss)
    if value > 1024 * 1024:
        return value / 1024 / 1024
    return value / 1024


def windows_rss_mb() -> float:
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return 0.0
    return counters.WorkingSetSize / 1024 / 1024


def gpu_alloc_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 / 1024


def gpu_reserved_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved() / 1024 / 1024


@contextmanager
def timed_stage(profile: PdfProfile, name: str) -> Iterator[None]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    rss_before = process_rss_mb()
    gpu_before = gpu_alloc_mb()
    started = time.perf_counter()

    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    rss_after = process_rss_mb()
    gpu_after = gpu_alloc_mb()

    profile.stages[name] = StageStats(
        seconds=elapsed,
        rss_delta_mb=rss_after - rss_before,
        gpu_alloc_delta_mb=gpu_after - gpu_before,
    )
    profile.peak_rss_mb = max(profile.peak_rss_mb, rss_after)
    profile.peak_gpu_alloc_mb = max(profile.peak_gpu_alloc_mb, gpu_after)


def find_pdfs(data_dir: Path, limit: int | None) -> list[Path]:
    pdfs = sorted(data_dir.rglob("*.pdf"))
    if limit is not None:
        return pdfs[:limit]
    return pdfs


def encode_chunks(
    model: LocalEmbeddingModel,
    chunks,
    batch_size: int,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for batch_index, batch in enumerate(batched(chunks, batch_size), start=1):
        texts = [chunk.text for chunk in batch]
        started = time.perf_counter()
        vectors = model.encode_documents(texts, batch_size=batch_size)
        elapsed = time.perf_counter() - started
        embeddings.extend(vectors)
        print(
            f"      embedding batch {batch_index}: "
            f"{len(batch)} chunks, {elapsed:.2f}s, "
            f"gpu_alloc={gpu_alloc_mb():.0f}MB, gpu_reserved={gpu_reserved_mb():.0f}MB"
        )
    return embeddings


def profile_pdf(
    pdf_path: Path,
    model: LocalEmbeddingModel,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    write_db: bool,
    postgres_dsn: str | None,
) -> PdfProfile:
    profile = PdfProfile(
        filename=pdf_path.name,
        size_mb=pdf_path.stat().st_size / 1024 / 1024,
        peak_rss_mb=process_rss_mb(),
        peak_gpu_alloc_mb=gpu_alloc_mb(),
    )
    total_started = time.perf_counter()
    print(f"\n== {pdf_path.name} ({profile.size_mb:.2f} MB) ==")

    with timed_stage(profile, "hash"):
        file_sha = sha256_file(pdf_path)
        document_id = file_sha[:32]

    with timed_stage(profile, "extract_text"):
        pages = extract_pages(pdf_path)
    profile.pages = len(pages)

    with timed_stage(profile, "chunk"):
        chunks = chunk_pages(document_id, pages, chunk_size, chunk_overlap)
    profile.chunks = len(chunks)

    print(f"   pages={profile.pages}, chunks={profile.chunks}")

    with timed_stage(profile, "embedding"):
        embeddings = encode_chunks(model, chunks, batch_size)

    if write_db:
        if not postgres_dsn:
            raise SystemExit("POSTGRES_DSN is required when --write-db is used.")
        with timed_stage(profile, "write_db"):
            with connect(postgres_dsn) as conn:
                init_schema(conn, vector_dim=model.dimension, recreate=False)
                register_vector(conn)
                upsert_document(conn, document_id, pdf_path, file_sha, len(pages))
                replace_chunks(conn, chunks, embeddings)
                conn.commit()

    profile.total_seconds = time.perf_counter() - total_started
    print_profile(profile)
    return profile


def print_stage(name: str, stats: StageStats) -> None:
    print(
        f"   {name:<12} "
        f"{stats.seconds:>8.2f}s  "
        f"rss_delta={stats.rss_delta_mb:>8.1f}MB  "
        f"gpu_delta={stats.gpu_alloc_delta_mb:>8.1f}MB"
    )


def print_profile(profile: PdfProfile) -> None:
    for name in ("hash", "extract_text", "chunk", "embedding", "write_db"):
        if name in profile.stages:
            print_stage(name, profile.stages[name])
    print(
        f"   total        {profile.total_seconds:>8.2f}s  "
        f"peak_rss={profile.peak_rss_mb:>8.1f}MB  "
        f"peak_gpu_alloc={profile.peak_gpu_alloc_mb:>8.1f}MB"
    )


def print_summary(profiles: list[PdfProfile]) -> None:
    if not profiles:
        return
    print("\n== Summary ==")
    stage_names = ["hash", "extract_text", "chunk", "embedding", "write_db"]
    for stage in stage_names:
        values = [profile.stages[stage].seconds for profile in profiles if stage in profile.stages]
        if values:
            print(
                f"{stage:<12} total={sum(values):>8.2f}s  "
                f"avg={statistics.mean(values):>8.2f}s"
            )
    print(f"{'all':<12} total={sum(profile.total_seconds for profile in profiles):>8.2f}s")


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Profile PDF ingestion stages without using the Web UI.")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "Database"))
    parser.add_argument("--pdf", default=None, help="Profile one specific PDF path.")
    parser.add_argument("--limit", type=int, default=1, help="Number of PDFs to profile when --pdf is not set.")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "1200")))
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "200")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "4")))
    parser.add_argument("--write-db", action="store_true", help="Also measure PostgreSQL write time.")
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN"))
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is not available. Pass --allow-cpu if you want to profile CPU embedding.")

    if args.pdf:
        pdfs = [Path(args.pdf)]
    else:
        pdfs = find_pdfs(Path(args.data_dir), args.limit)
    if not pdfs:
        raise SystemExit("No PDF files found.")

    print(f"Python PID: {os.getpid()}")
    print(f"Embedding model: {args.model}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Initial RSS: {process_rss_mb():.1f} MB")

    model_load_profile = PdfProfile(filename="model_load", size_mb=0)
    with timed_stage(model_load_profile, "load_model"):
        model = load_embedding_model(args.model)
    print_stage("load_model", model_load_profile.stages["load_model"])
    print(f"Embedding device: {model.device}")
    print(f"Embedding dimension: {model.dimension}")
    print(f"Half precision: {model.half_precision}")
    print(f"Max sequence length: {model.model.max_seq_length}")
    print(f"After model load RSS: {process_rss_mb():.1f} MB")
    if torch.cuda.is_available():
        print(f"After model load GPU reserved: {gpu_reserved_mb():.1f} MB")

    profiles = [
        profile_pdf(
            pdf_path=pdf,
            model=model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            write_db=args.write_db,
            postgres_dsn=args.postgres_dsn,
        )
        for pdf in pdfs
    ]
    print_summary(profiles)


if __name__ == "__main__":
    main()
