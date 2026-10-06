#!/usr/bin/env python3
"""Export clip-local aligned transcripts from an audit batch manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def build_records(
    batch_manifest: dict[str, Any],
    vlm_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items = batch_manifest.get("items") or []
    by_identity = {
        (int(item["ordinal"]), item["uid"]): item
        for item in items
    }
    if len(by_identity) != len(items):
        raise ValueError("batch manifest contains duplicate ordinal/uid identities")
    records = []
    for row in vlm_manifest:
        identity = (int(row["ordinal"]), row["uid"])
        if identity not in by_identity:
            raise ValueError(f"VLM row absent from batch manifest: {identity}")
        item = by_identity[identity]
        segments = []
        for segment in item.get("aligned_transcript") or []:
            start = segment.get("clip_start", segment.get("start"))
            end = segment.get("clip_end", segment.get("end"))
            if start is None or end is None:
                raise ValueError(
                    f"transcript segment lacks timestamps: {row['item_id']}"
                )
            segments.append(
                {
                    "start": max(0.0, float(start)),
                    "end": max(0.0, float(end)),
                    "text": str(segment.get("text", "")).strip(),
                }
            )
        records.append(
            {
                "item_id": row["item_id"],
                "uid": row["uid"],
                "ordinal": int(row["ordinal"]),
                "segments": segments,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--vlm-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = build_records(
        json.loads(args.batch_manifest.read_text()),
        load_jsonl(args.vlm_manifest),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "clip_local_aligned_transcripts",
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
