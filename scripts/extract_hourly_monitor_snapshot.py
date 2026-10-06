#!/usr/bin/env python3
"""Freeze one exact hourly-monitor JSONL record for a manual audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def extract(path: Path, checked_at: str) -> bytes:
    matches: list[bytes] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if isinstance(row, dict) and row.get("checked_at") == checked_at:
                matches.append(line.rstrip(b"\r\n") + b"\n")
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one snapshot at {checked_at!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-jsonl", type=Path, required=True)
    parser.add_argument("--checked-at", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    payload = extract(args.snapshots_jsonl, args.checked_at)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    print(
        json.dumps(
            {
                "checked_at": args.checked_at,
                "out": str(args.out),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
