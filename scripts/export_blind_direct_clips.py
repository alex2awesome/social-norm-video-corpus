#!/usr/bin/env python3
"""Copy selected audit clips to opaque filenames for blind direct review."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()

    wanted = set(args.indices)
    rows = [
        json.loads(line)
        for line in args.semantic_manifest.read_text().splitlines()
        if line.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[int] = []
    for row in rows:
        audit_index = int(row["audit_index"])
        if audit_index not in wanted:
            continue
        shutil.copy2(row["source_clip"], args.output_dir / f"{audit_index:04d}.mp4")
        copied.append(audit_index)

    missing = sorted(wanted - set(copied))
    if missing:
        raise SystemExit(f"missing indices: {missing}")
    print(json.dumps({"copied_indices": sorted(copied)}))


if __name__ == "__main__":
    main()
