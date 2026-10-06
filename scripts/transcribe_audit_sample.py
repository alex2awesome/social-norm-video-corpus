#!/usr/bin/env python3
"""Append local faster-whisper transcripts to an audiovisual audit artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-type")
    parser.add_argument("--uid", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        artifact = json.loads(args.output.read_text())
    else:
        artifact = {"model": args.model, "records": []}

    completed = {record["uid"] for record in artifact["records"]}
    requested = set(args.uid)
    media = sorted(args.media_dir.glob("*.mp4"))
    if requested:
        media = [path for path in media if path.stem in requested]

    compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)
    for position, path in enumerate(media, start=1):
        uid = path.stem
        if uid in completed:
            continue
        print(f"[{position}/{len(media)}] {uid}", flush=True)
        segments, info = model.transcribe(
            str(path),
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        record = {
            "uid": uid,
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": [
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                }
                for segment in segments
            ],
        }
        artifact["records"].append(record)
        artifact["records"].sort(key=lambda item: item["uid"])
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")


if __name__ == "__main__":
    main()
