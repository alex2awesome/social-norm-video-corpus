#!/usr/bin/env python3
"""Convert an opaque storyboard manifest into a VLM runner manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BLIND_FIELDS = {
    "audit_index",
    "candidate_id",
    "sheet_path",
    "sheet_sha256",
}


def build(path: Path, frame_count: int) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if any(set(row) != BLIND_FIELDS for row in rows):
        raise ValueError("blind manifest contains missing or semantic fields")
    expected = set(range(len(rows)))
    if {int(row["audit_index"]) for row in rows} != expected:
        raise ValueError("blind audit_index coverage is not contiguous")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    return [
        {
            "audit_index": int(row["audit_index"]),
            "candidate_id": str(row["candidate_id"]),
            "item_id": str(row["candidate_id"]),
            "pillar": "instructional",
            "sheet_path": str(row["sheet_path"]),
            "sheet_sha256": str(row["sheet_sha256"]),
            "frame_count": frame_count,
        }
        for row in sorted(rows, key=lambda value: int(value["audit_index"]))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=36)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    rows = build(args.blind_manifest, args.frame_count)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    print(
        json.dumps(
            {
                "kind": "blind_vlm_manifest",
                "items": len(rows),
                "semantic_metadata_emitted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
