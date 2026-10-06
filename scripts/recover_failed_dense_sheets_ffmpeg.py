#!/usr/bin/env python3
"""Recover OpenCV-seek failures as uniform ffmpeg-decoded compact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def failed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("error")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duration(path: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    value = float(result.stdout.strip())
    if value <= 0:
        raise ValueError(f"invalid duration: {value}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--columns", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    if args.frames < 1 or args.frames % args.columns:
        raise SystemExit("--frames must be divisible by --columns")
    args.out.mkdir(parents=True)
    sheets = args.out / "sheets"
    sheets.mkdir()

    output = []
    for row in failed_rows(load_jsonl(args.semantic_selection)):
        source = Path(row["source_path"])
        target = sheets / f"{row['candidate_id']}.jpg"
        try:
            seconds = duration(source, args.ffprobe)
            rate = args.frames / seconds
            rows = args.frames // args.columns
            subprocess.run(
                [
                    args.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(source),
                    "-vf",
                    (
                        f"fps={rate:.9f},"
                        "scale=320:180:force_original_aspect_ratio=decrease,"
                        "pad=320:180:(ow-iw)/2:(oh-ih)/2:black,"
                        f"tile={args.columns}x{rows}"
                    ),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "4",
                    str(target),
                ],
                check=True,
            )
            output.append(
                {
                    "audit_index": row["audit_index"],
                    "candidate_id": row["candidate_id"],
                    "item_id": row["candidate_id"],
                    "sheet_path": str(target),
                    "sheet_sha256": sha256(target),
                    "frame_count": args.frames,
                    "sampling_fps": rate,
                    "recovery_method": "sequential_ffmpeg_uniform_tile",
                    "source_error": row["error"],
                    "error": None,
                }
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            output.append(
                {
                    "audit_index": row["audit_index"],
                    "candidate_id": row["candidate_id"],
                    "item_id": row["candidate_id"],
                    "sheet_path": None,
                    "sheet_sha256": None,
                    "frame_count": 0,
                    "sampling_fps": None,
                    "recovery_method": "sequential_ffmpeg_uniform_tile",
                    "source_error": row["error"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    manifest = args.out / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output)
    )
    summary = {
        "kind": "failed_dense_sheet_ffmpeg_recovery",
        "attempted": len(output),
        "recovered": sum(row["error"] is None for row in output),
        "still_failed": sum(row["error"] is not None for row in output),
        "manifest_sha256": sha256(manifest),
        "semantic_metadata_emitted": False,
        "corpus_mutated": False,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
