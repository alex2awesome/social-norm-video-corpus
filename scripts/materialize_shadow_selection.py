#!/usr/bin/env python3
"""Join a frozen item selection back to its immutable corpus manifest."""

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


def materialize(
    manifest: list[dict[str, Any]],
    selection: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {row.get("item_id"): row for row in manifest if row.get("item_id")}
    if len(by_id) != len(manifest):
        raise ValueError("manifest item_id values must be present and unique")
    seen = set()
    result = []
    for selected in selection:
        item_id = selected.get("item_id")
        if not item_id or item_id in seen:
            raise ValueError("selection item_id values must be present and unique")
        if item_id not in by_id:
            raise ValueError(f"selected item absent from manifest: {item_id}")
        seen.add(item_id)
        row = dict(by_id[item_id])
        row["shadow_selection"] = {
            key: value
            for key, value in selected.items()
            if key not in row or row.get(key) != value
        }
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    rows = materialize(read_jsonl(args.manifest), read_jsonl(args.selection))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"items": len(rows), "out": str(args.out)}))


if __name__ == "__main__":
    main()
