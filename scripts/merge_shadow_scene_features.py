#!/usr/bin/env python3
"""Join append-only shadow feature JSONL files by item_id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_rows(base_rows: list[dict], additions: list[list[dict]]) -> list[dict]:
    """Merge complementary scorer sections without erasing prior results.

    A scorer writes unrequested sections as ``null``.  Those placeholders must
    not clobber a section produced by an earlier scorer.
    """
    rows = {row["item_id"]: dict(row) for row in base_rows}
    for addition_rows in additions:
        for addition in addition_rows:
            item_id = addition["item_id"]
            if item_id not in rows:
                continue
            for key, value in addition.items():
                if key in {
                    "item_id",
                    "uid",
                    "pillar",
                    "gold_scene_visible",
                    "gold_social_scene_visible",
                    "gold_label_matched_visible",
                    "error",
                } or value is None:
                    continue
                rows[item_id][key] = value
    return list(rows.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--add", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = merge_rows(
        load_jsonl(args.base),
        [load_jsonl(path) for path in args.add],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
