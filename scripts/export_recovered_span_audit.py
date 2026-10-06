#!/usr/bin/env python3
"""Export temporal frames for shadow-recovered instructional spans.

Reads the JSONL produced by audit_recover_instructional_spans.py and samples
only a requested confidence class. It never writes corpus clips or metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any


RAW_EXTENSIONS = {".mp4", ".mkv", ".webm", ".m4a", ".mov"}


def extract_frames(
    ffmpeg: str,
    raw: Path,
    start: float,
    end: float,
    frames_dir: Path,
    ordinal: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    duration = end - start
    frame_count = min(max_frames, max(3, math.ceil(duration)))
    frames = []
    for index in range(frame_count):
        offset = duration * (index + 0.5) / frame_count
        timestamp = start + offset
        name = f"{ordinal:04d}_f{index:02d}.jpg"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(raw),
                "-frames:v",
                "1",
                "-vf",
                "scale=768:-2",
                "-q:v",
                "3",
                "-y",
                str(frames_dir / name),
            ],
            check=True,
        )
        frames.append(
            {
                "frame_index": index,
                "source_timestamp": round(timestamp, 3),
                "clip_timestamp": round(offset, 3),
                "path": f"frames/{name}",
            }
        )
    return frames


def aligned_transcript(path: Path, start: float, end: float) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    result = []
    for segment in payload.get("segments") or []:
        try:
            segment_start = float(segment["start"])
            segment_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if segment_end < start - 1 or segment_start > end + 1:
            continue
        result.append(
            {
                "start": round(segment_start, 3),
                "end": round(segment_end, 3),
                "clip_start": round(segment_start - start, 3),
                "clip_end": round(segment_end - start, 3),
                "text": str(segment.get("text") or "").strip(),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confidence", default="exact_unique")
    parser.add_argument("--count", type=int, default=24, help="0 exports every matching record")
    parser.add_argument("--seed", default="recovered-span-audit-v1")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir()
    rows = []
    for line in args.candidates.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("confidence") == args.confidence and row.get("best"):
            rows.append(row)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.count:
        rows = rows[: args.count]

    root = args.project_root.resolve()
    raw_by_uid = {
        path.stem: path
        for path in (root / "data" / "raw_video").iterdir()
        if path.suffix.lower() in RAW_EXTENSIONS
    }
    records = []
    for ordinal, row in enumerate(rows):
        raw = raw_by_uid[row["uid"]]
        best = row["best"]
        start, end = float(best["start_sec"]), float(best["end_sec"])
        try:
            frames = extract_frames(
                args.ffmpeg, raw, start, end, frames_dir, ordinal, args.max_frames
            )
            error = None
        except Exception as exc:
            frames, error = [], str(exc)
        frame_hash = hashlib.sha256(
            json.dumps(frames, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        records.append(
            {
                **row,
                "ordinal": ordinal,
                "raw_path": str(raw.relative_to(root)),
                "frames": frames,
                "frame_manifest_sha256": frame_hash,
                "frame_error": error,
                "aligned_transcript": aligned_transcript(
                    root / "data" / "transcripts" / f"{row['uid']}.json", start, end
                ),
            }
        )
    manifest = {
        "audit_type": "recovered_instructional_span_v1",
        "confidence": args.confidence,
        "seed": args.seed,
        "eligible": sum(1 for line in args.candidates.read_text().splitlines() if line.strip() and json.loads(line).get("confidence") == args.confidence),
        "items": records,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"eligible": manifest["eligible"], "items": len(records), "frames": sum(len(row["frames"]) for row in records)}))


if __name__ == "__main__":
    main()
