#!/usr/bin/env python3
"""Export exact commentary windows for cheap visual shadow mechanisms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_rows(
    windows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    proxy_dir: Path,
) -> list[dict[str, Any]]:
    semantic = {int(row["audit_index"]): row for row in semantic_rows}
    rows = []
    seen: set[int] = set()
    for window in windows:
        index = int(window["audit_index"])
        if index in seen:
            raise ValueError(f"duplicate audit index: {index}")
        seen.add(index)
        if index not in semantic:
            raise ValueError(f"missing semantic row: {index}")
        if window["manual_gold"] not in {"positive", "negative"}:
            raise ValueError(f"invalid manual gold: {index}")
        source = semantic[index]
        clip = (proxy_dir / f"{source['candidate_id']}.mp4").resolve()
        if not clip.is_file():
            raise FileNotFoundError(clip)
        gold = window["manual_gold"] == "positive"
        rows.append(
            {
                "audit_index": index,
                "item_id": source["candidate_id"],
                "uid": source["uid"],
                "pillar": "commentary",
                "source_clip": str(clip),
                "media_start_sec": float(window["start_sec"]),
                "media_end_sec": float(window["end_sec"]),
                "gold_scene_visible": gold,
                "gold_social_scene_visible": gold,
                "gold_label_matched_visible": gold,
                "gold_usable": gold,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--semantic", required=True, type=Path)
    parser.add_argument("--proxy-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    rows = build_rows(
        read_jsonl(args.windows),
        read_jsonl(args.semantic),
        args.proxy_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
