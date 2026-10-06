#!/usr/bin/env python3
"""Merge disjoint opaque VLM manifests with exact coverage checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def merge(paths: list[Path], expected: int) -> list[dict[str, Any]]:
    rows = [row for path in paths for row in load_jsonl(path)]
    ids = [row["item_id"] for row in rows]
    indices = [int(row["audit_index"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item_id across manifests")
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate audit_index across manifests")
    if len(rows) != expected or set(indices) != set(range(expected)):
        raise ValueError("manifests do not exactly cover expected audit indices")
    if any(row.get("error") for row in rows):
        raise ValueError("manifest contains unresolved errors")
    return sorted(rows, key=lambda row: int(row["audit_index"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    rows = merge(args.manifest, args.expected)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    print(
        json.dumps(
            {
                "items": len(rows),
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
