#!/usr/bin/env python3
"""Cut statement-centered commentary windows for blind visual review."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def statement_window(
    start: float,
    end: float,
    duration: float,
    before: float,
    after: float,
) -> tuple[float, float]:
    if not (0 <= start < end and duration > 0):
        raise ValueError("invalid statement timing")
    return max(0.0, start - before), min(duration, end + after)


def overlapping_segments(
    transcript: dict[str, Any], start: float, end: float
) -> list[dict[str, Any]]:
    return [
        {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": segment.get("text"),
        }
        for segment in transcript.get("segments") or []
        if float(segment.get("end", 0)) >= start
        and float(segment.get("start", 0)) <= end
    ]


def probe_duration(ffprobe: str, media: Path) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(media),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def cut_window(
    ffmpeg: str, media: Path, target: Path, start: float, end: float
) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(media),
            "-t",
            f"{end - start:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-y",
            str(target),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--before-sec", type=float, default=12.0)
    parser.add_argument("--after-sec", type=float, default=12.0)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    args.out.mkdir(parents=True)
    clips_dir = args.out / "clips"
    clips_dir.mkdir()
    blind, packets = [], []
    for source in read_jsonl(args.selection):
        index = int(source["audit_index"])
        uid = str(source["uid"])
        metadata_path = args.root / "data" / "discussion" / f"{uid}.json"
        transcript_path = args.root / "data" / "transcripts" / f"{uid}.json"
        metadata = json.loads(metadata_path.read_text())
        statements = metadata.get("statements") or []
        if not statements:
            raise ValueError(f"{uid}: no commentary statements")
        statement = statements[0]
        media = Path(source["clip"])
        duration = probe_duration(args.ffprobe, media)
        start, end = statement_window(
            float(statement["start"]),
            float(statement["end"]),
            duration,
            args.before_sec,
            args.after_sec,
        )
        target = clips_dir / f"{index:02d}_{uid}.mp4"
        cut_window(args.ffmpeg, media, target, start, end)
        item_id = f"commentary_statement:{uid}:0"
        blind.append(
            {
                "audit_index": index,
                "item_id": item_id,
                "uid": uid,
                "clip": str(target.resolve()),
            }
        )
        transcript = json.loads(transcript_path.read_text())
        packets.append(
            {
                "audit_index": index,
                "item_id": item_id,
                "uid": uid,
                "title": source.get("title"),
                "query": source.get("query"),
                "query_source": source.get("query_source"),
                "statement": statement,
                "window_start_sec": start,
                "window_end_sec": end,
                "source_duration": duration,
                "transcript_segments": overlapping_segments(transcript, start, end),
                "source_media": str(media),
                "window_media": str(target.resolve()),
            }
        )
    for name, rows in (
        ("blind_selection.jsonl", blind),
        ("semantic_packets.jsonl", packets),
    ):
        (args.out / name).write_text(
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
        )
    print(
        json.dumps(
            {
                "items": len(blind),
                "window_seconds": sum(
                    row["window_end_sec"] - row["window_start_sec"] for row in packets
                ),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
