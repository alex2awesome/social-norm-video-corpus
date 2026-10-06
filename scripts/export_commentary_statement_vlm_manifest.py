#!/usr/bin/env python3
"""Convert statement-window packets into a VLM-ready commentary manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def convert(packet: dict[str, Any]) -> dict[str, Any]:
    offset = float(packet["window_start_sec"])
    aligned = [
        {
            "start": max(0.0, float(segment["start"]) - offset),
            "end": max(0.0, float(segment["end"]) - offset),
            "text": segment.get("text"),
        }
        for segment in packet.get("transcript_segments") or []
    ]
    statement = packet["statement"]
    return {
        "ordinal": int(packet["audit_index"]),
        "item_id": packet["item_id"],
        "uid": packet["uid"],
        "pillar": "commentary",
        "source_clip": packet["window_media"],
        "duration_hint": float(packet["window_end_sec"])
        - float(packet["window_start_sec"]),
        "norm": statement.get("norm"),
        "explanation": statement.get("quote"),
        "aligned_transcript": aligned,
        "statement_signal": statement.get("signal"),
        "policy": "shadow_vlm_confirmation_no_keep_reject",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    rows = [convert(packet) for packet in read_jsonl(args.packets)]
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    )
    print(json.dumps({"items": len(rows), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
