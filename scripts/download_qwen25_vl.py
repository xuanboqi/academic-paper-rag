from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen2.5-VL-3B-Instruct to the local models directory.")
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--resume", action="store_true", help="Keep and reuse partially downloaded files.")
    parser.add_argument("--max-workers", type=int, default=1, help="Lower values are more stable on weak network connections.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id} -> {output_dir.resolve()}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
        resume_download=args.resume,
        max_workers=args.max_workers,
    )
    print("Done.")


if __name__ == "__main__":
    main()
