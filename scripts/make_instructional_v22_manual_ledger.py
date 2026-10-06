#!/usr/bin/env python3
"""Create a complete blind manual-ledger template for instructional V22."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "audit_index",
    "candidate_id",
    "item_id",
    "uid",
    "visual_form",
    "visual_demo",
    "semantic_alignment",
    "complete_demo",
    "usable_demo",
    "note",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def template_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output = []
    for row in sorted(manifest, key=lambda value: int(value["audit_index"])):
        item_id = str(row["item_id"])
        if item_id in seen:
            raise ValueError(f"duplicate item_id: {item_id}")
        seen.add(item_id)
        output.append(
            {
                "audit_index": row["audit_index"],
                "candidate_id": row["candidate_id"],
                "item_id": item_id,
                "uid": row["uid"],
                "visual_form": "",
                "visual_demo": "",
                "semantic_alignment": "",
                "complete_demo": "",
                "usable_demo": "",
                "note": "",
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    rows = template_rows(read_jsonl(args.manifest))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
