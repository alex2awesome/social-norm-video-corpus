#!/usr/bin/env python3
"""Materialize a new manual-review version from explicit adjudications.

The base review is never modified. Each adjudication names one item, carries a
complete replacement row, and records why evidence available after the blind
review changed the judgment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_adjudications(
    base: list[dict[str, Any]],
    adjudications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(row["item_id"]): row for row in base}
    if len(by_id) != len(base):
        raise ValueError("duplicate base item_id")
    seen = set()
    for adjudication in adjudications:
        item_id = str(adjudication["item_id"])
        if item_id in seen:
            raise ValueError(f"duplicate adjudication: {item_id}")
        seen.add(item_id)
        if item_id not in by_id:
            raise ValueError(f"adjudication absent from base: {item_id}")
        replacement = adjudication.get("replacement")
        if not isinstance(replacement, dict):
            raise ValueError(f"{item_id}: replacement must be an object")
        if replacement.get("item_id") != item_id:
            raise ValueError(f"{item_id}: replacement identity mismatch")
        if set(replacement) != set(by_id[item_id]):
            raise ValueError(f"{item_id}: replacement schema mismatch")
        if not str(adjudication.get("reason") or "").strip():
            raise ValueError(f"{item_id}: missing adjudication reason")
        by_id[item_id] = replacement
    return [by_id[str(row["item_id"])] for row in base]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    base = read_jsonl(args.base)
    adjudications = read_jsonl(args.adjudications)
    output = apply_adjudications(base, adjudications)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output)
    )
    summary = {
        "schema_version": 1,
        "kind": "manual_review_adjudication_materialization",
        "base": str(args.base),
        "base_sha256": sha256(args.base),
        "adjudications": str(args.adjudications),
        "adjudications_sha256": sha256(args.adjudications),
        "adjudicated_items": len(adjudications),
        "items": len(output),
        "out": str(args.out),
        "out_sha256": sha256(args.out),
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
