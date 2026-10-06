#!/usr/bin/env python3
"""Align successfully rendered storyboards with their sealed semantic rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def align(
    storyboard_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    semantics = {str(row["item_id"]): row for row in semantic_rows}
    if len(semantics) != len(semantic_rows):
        raise ValueError("duplicate semantic item_id")
    rendered = [
        row for row in storyboard_rows
        if not row.get("error") and row.get("sheet_path") and row.get("sheet_sha256")
    ]
    item_ids = [str(row["item_id"]) for row in rendered]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("duplicate storyboard item_id")
    missing = sorted(set(item_ids) - semantics.keys())
    if missing:
        raise ValueError(f"storyboards lack semantic rows: {missing[:3]}")
    aligned_semantics = [semantics[item_id] for item_id in item_ids]
    return rendered, aligned_semantics, {
        "input_storyboards": len(storyboard_rows),
        "input_semantics": len(semantic_rows),
        "eligible": len(rendered),
        "render_failures_retained_outside_model_input": len(storyboard_rows) - len(rendered),
        "corpus_mutated": False,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storyboard-manifest", type=Path, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--out-storyboards", type=Path, required=True)
    parser.add_argument("--out-semantics", type=Path, required=True)
    args = parser.parse_args()
    storyboards, semantics, summary = align(
        read_jsonl(args.storyboard_manifest), read_jsonl(args.semantic_manifest)
    )
    write_jsonl(args.out_storyboards, storyboards)
    write_jsonl(args.out_semantics, semantics)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
