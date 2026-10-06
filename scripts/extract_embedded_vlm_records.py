#!/usr/bin/env python3
"""Extract one embedded VLM record per selected audit row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract(path: Path, field: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    output: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        record = row.get(field)
        if not isinstance(record, dict):
            raise ValueError(f"audit_index {row.get('audit_index')}: missing {field}")
        if record.get("item_id") != row.get("item_id"):
            raise ValueError(
                f"audit_index {row.get('audit_index')}: embedded item_id mismatch"
            )
        item_id = str(record["item_id"])
        if item_id in seen:
            raise ValueError(f"duplicate item_id: {item_id}")
        seen.add(item_id)
        output.append(record)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    rows = extract(args.selection, args.field)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    print(json.dumps({"field": args.field, "items": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
