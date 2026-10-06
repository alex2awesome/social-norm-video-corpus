#!/usr/bin/env python3
"""Build title/query/visual-blind text packets for instructional sociality audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORBIDDEN = {
    "item_id", "uid", "title", "category", "found_by_query", "query_source",
    "genre", "source_platform", "clip_path", "source_clip",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indices = [row.get("audit_index") for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("instructional source indices are incomplete or reordered")
    output = []
    for row in rows:
        packet = {
            "blind_id": f"instructional-social-v5-{int(row['audit_index']):04d}",
            "audit_index": int(row["audit_index"]),
            "proposed_norm": str(row.get("norm") or ""),
            "proposed_polarity": str(row.get("polarity") or ""),
            "start_quote": str(row.get("start_quote") or ""),
            "end_quote": str(row.get("end_quote") or ""),
            "explanation": str(row.get("explanation") or ""),
        }
        if set(packet) & FORBIDDEN:
            raise AssertionError("instructional blind packet leaks retrieval or visual metadata")
        output.append(packet)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    for row in build(read_jsonl(args.selection)):
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
