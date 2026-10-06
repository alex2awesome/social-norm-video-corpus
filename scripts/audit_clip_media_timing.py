#!/usr/bin/env python3
"""Non-destructively scan corpus clips for delayed or missing video evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from src.media_integrity import probe_media_timing, timing_flags
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.media_integrity import probe_media_timing, timing_flags


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}


def stable_sample(paths: list[Path], limit: int | None, seed: str) -> list[Path]:
    paths = sorted(paths)
    if limit is None or limit >= len(paths):
        return paths
    rng = random.Random(seed)
    return sorted(rng.sample(paths, limit))


def scan_one(
    path: Path,
    root: Path,
    pillar: str,
    ffprobe: str,
    max_gap: float,
    min_coverage: float,
) -> dict:
    try:
        timing = probe_media_timing(path, ffprobe)
        flags = timing_flags(
            timing,
            max_leading_video_gap_sec=max_gap,
            min_video_coverage_ratio=min_coverage,
        )
        error = None
    except Exception as exc:
        timing = None
        flags = ["probe_error"]
        error = f"{type(exc).__name__}: {exc}"
    return {
        "pillar": pillar,
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "path_key": hashlib.sha256(str(path).encode()).hexdigest()[:16],
        "size_bytes": path.stat().st_size if path.exists() else None,
        "timing": timing,
        "flags": flags,
        "flagged": bool(flags),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pillar", required=True)
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", default="media-timing-audit-v1")
    parser.add_argument("--max-leading-video-gap-sec", type=float, default=0.5)
    parser.add_argument("--min-video-coverage-ratio", type=float, default=0.8)
    args = parser.parse_args()

    patterns = args.glob or ["**/*"]
    found = {
        path
        for pattern in patterns
        for path in args.root.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    }
    selected = stable_sample(list(found), args.limit, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                scan_one,
                path,
                args.root,
                args.pillar,
                args.ffprobe,
                args.max_leading_video_gap_sec,
                args.min_video_coverage_ratio,
            ): path
            for path in selected
        }
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            if index % 100 == 0 or index == len(selected):
                print(f"{index}/{len(selected)}", flush=True)
    records.sort(key=lambda row: row["relative_path"])
    with args.out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "pillar": args.pillar,
        "root": str(args.root),
        "population_files": len(found),
        "sampled_files": len(records),
        "flagged_files": sum(row["flagged"] for row in records),
        "probe_errors": sum(row["error"] is not None for row in records),
        "flag_counts": {
            flag: sum(flag in row["flags"] for row in records)
            for flag in sorted({flag for row in records for flag in row["flags"]})
        },
        "out": str(args.out),
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
