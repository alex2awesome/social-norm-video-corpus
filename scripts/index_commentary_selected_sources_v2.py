#!/usr/bin/env python3
"""Index retained full videos for manually accepted commentary text labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".mov", ".m4v")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def locate(video_dir: Path, uid: str) -> Path:
    matches = [video_dir / f"{uid}{suffix}" for suffix in VIDEO_SUFFIXES]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"{uid}: expected exactly one retained source, found {len(matches)}")
    return matches[0]


def probe_duration(ffprobe: str, path: Path) -> float:
    value = subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True).strip()
    duration = float(value)
    if duration <= 0:
        raise ValueError("nonpositive media duration")
    return duration


def index(
    labels: list[dict[str, Any]], video_dir: Path, ffprobe: str, max_duration: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    uids = [str(row.get("uid") or "") for row in labels]
    if not uids or any(not uid for uid in uids) or len(uids) != len(set(uids)):
        raise ValueError("labels have missing or duplicate uid")
    successes, failures = [], []
    for uid in uids:
        try:
            path = locate(video_dir, uid)
            duration = probe_duration(ffprobe, path)
            if duration > max_duration:
                raise ValueError(
                    f"duration {duration:.3f}s exceeds audit limit {max_duration:.3f}s"
                )
            successes.append({
                "uid": uid,
                "source_path": str(path.resolve()),
                "duration_sec": duration,
                "policy": "retained_source_index_only_no_media_copy",
            })
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            failures.append({"uid": uid, "error": f"{type(exc).__name__}: {exc}"})
    return successes, failures


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-duration", type=float, default=1800)
    args = parser.parse_args()
    for path in (args.out, args.failures, args.summary):
        if path.exists():
            raise FileExistsError(path)
    rows, failures = index(
        read_jsonl(args.labels), args.video_dir, args.ffprobe, args.max_duration
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    write_jsonl(args.failures, failures)
    summary = {
        "kind": "commentary_retained_source_index_v2",
        "accepted_text_labels": len(rows) + len(failures),
        "indexed_sources": len(rows),
        "failed_sources": len(failures),
        "media_copied": False,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
        "index_sha256": sha256(args.out),
        "failures_sha256": sha256(args.failures),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    # A partially indexable manually accepted cohort may proceed with explicit
    # failures preserved, but an empty index must stop the visual pipeline.
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
