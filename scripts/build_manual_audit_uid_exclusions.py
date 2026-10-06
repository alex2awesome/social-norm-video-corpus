#!/usr/bin/env python3
"""Collect source UIDs present in manual audit artifacts.

This is intentionally conservative: any source named by a manual review, gold,
or adjudication artifact is excluded from a fresh source-disjoint cohort.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


MANUAL_MARKERS = ("manual", "review", "gold", "adjudicat")
ITEM_ID = re.compile(r"^(?:instructional|witnessed|commentary):([^:]+):")
UID_KEYS = {"uid", "video_id", "source_uid"}
ITEM_KEYS = {"item_id", "candidate_id"}


def collect_from_value(value: Any, result: set[str]) -> None:
    if isinstance(value, dict):
        for key, member in value.items():
            if key in UID_KEYS and isinstance(member, str) and member.strip():
                result.add(member.strip())
            elif key in ITEM_KEYS and isinstance(member, str):
                match = ITEM_ID.match(member.strip())
                if match:
                    result.add(match.group(1))
            collect_from_value(member, result)
    elif isinstance(value, list):
        for member in value:
            collect_from_value(member, result)


def collect_from_file(path: Path, result: set[str]) -> bool:
    try:
        if path.suffix in {".tsv", ".csv"}:
            with path.open(newline="", errors="replace") as handle:
                values = list(csv.DictReader(
                    handle, delimiter="\t" if path.suffix == ".tsv" else ","
                ))
        elif path.suffix == ".jsonl":
            values = [
                json.loads(line)
                for line in path.read_text(errors="replace").splitlines()
                if line.strip()
            ]
        else:
            values = [json.loads(path.read_text(errors="replace"))]
    except (OSError, json.JSONDecodeError):
        return False
    for value in values:
        collect_from_value(value, result)
    return True


def manual_artifacts(roots: list[Path]) -> list[Path]:
    return sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".json", ".jsonl", ".tsv", ".csv"}
        and any(marker in path.name.lower() for marker in MANUAL_MARKERS)
    )


def json_artifacts(roots: list[Path]) -> list[Path]:
    """Return every JSON artifact for a maximally conservative exclusion."""
    return sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument(
        "--include",
        type=Path,
        action="append",
        default=[],
        help="Explicit JSON/JSONL audit artifact to parse regardless of filename.",
    )
    parser.add_argument(
        "--all-json",
        action="store_true",
        help=(
            "Parse every JSON/JSONL artifact under each root. This broad mode "
            "is intended for a wholly fresh source-disjoint benchmark."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")

    discovered = (
        json_artifacts(args.root)
        if args.all_json
        else manual_artifacts(args.root)
    )
    paths = sorted(set(discovered + args.include))
    uids: set[str] = set()
    parsed = 0
    for path in paths:
        parsed += collect_from_file(path, uids)
    lines = [f"{uid}\n" for uid in sorted(uids)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines))
    digest = hashlib.sha256("".join(lines).encode()).hexdigest()
    summary = {
        "schema_version": 1,
        "roots": [str(path.resolve()) for path in args.root],
        "candidate_files": len(paths),
        "all_json": args.all_json,
        "parsed_files": parsed,
        "uids": len(uids),
        "sha256": digest,
        "out": str(args.out.resolve()),
    }
    args.out.with_suffix(args.out.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
