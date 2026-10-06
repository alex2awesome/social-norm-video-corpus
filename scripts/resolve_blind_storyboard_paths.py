#!/usr/bin/env python3
"""Resolve relative blind-storyboard paths without changing semantic fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--storyboard-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")

    rows = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        path = Path(row["sheet_path"])
        if not path.is_absolute():
            row["sheet_path"] = str((args.storyboard_root / path).resolve())
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


if __name__ == "__main__":
    main()
