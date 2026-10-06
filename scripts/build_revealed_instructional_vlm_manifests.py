#!/usr/bin/env python3
"""Build VLM inputs from a revealed instructional audit selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SEMANTIC_FIELDS = (
    "item_id",
    "uid",
    "title",
    "category",
    "norm",
    "polarity",
    "start_quote",
    "end_quote",
    "explanation",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(
    selection: list[dict[str, Any]], storyboard_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    storyboards: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(selection, key=lambda value: int(value["audit_index"])):
        item_id = row["item_id"]
        if item_id in seen:
            raise ValueError(f"duplicate item_id: {item_id}")
        seen.add(item_id)
        storyboard = dict(row["storyboard"])
        sheet = Path(storyboard["sheet_path"])
        if not sheet.is_absolute():
            sheet = storyboard_root / sheet
        storyboard["sheet_path"] = str(sheet.resolve())
        storyboards.append(storyboard)
        metadata.append({key: row.get(key) for key in SEMANTIC_FIELDS})
    return storyboards, metadata


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--storyboard-root", type=Path, required=True)
    parser.add_argument("--storyboards-out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    args = parser.parse_args()
    storyboards, metadata = build(
        load_jsonl(args.selection), args.storyboard_root
    )
    write_jsonl(args.storyboards_out, storyboards)
    write_jsonl(args.metadata_out, metadata)
    print(json.dumps({"storyboards": len(storyboards), "metadata": len(metadata)}))


if __name__ == "__main__":
    main()
