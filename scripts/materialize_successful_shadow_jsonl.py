#!/usr/bin/env python3
"""Materialize one successful record per manifest item from an append-only log.

The raw log is never edited.  If an item has multiple successful attempts, the
last successful record wins and duplication is reported in a sidecar summary.
Items with only failed attempts remain explicit in the summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize(manifest: list[dict], attempts: list[dict]) -> tuple[list[dict], dict]:
    manifest_ids = [row["item_id"] for row in manifest]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("manifest contains duplicate item_id values")
    successful: dict[str, dict] = {}
    attempt_counts = Counter()
    successful_counts = Counter()
    error_only: set[str] = set()
    unexpected: set[str] = set()
    manifest_set = set(manifest_ids)
    for row in attempts:
        item_id = row["item_id"]
        attempt_counts[item_id] += 1
        if item_id not in manifest_set:
            unexpected.add(item_id)
            continue
        if row.get("result") is not None and row.get("error") is None:
            successful[item_id] = row
            successful_counts[item_id] += 1
            error_only.discard(item_id)
        elif item_id not in successful:
            error_only.add(item_id)
    rows = [successful[item_id] for item_id in manifest_ids if item_id in successful]
    missing = [item_id for item_id in manifest_ids if item_id not in successful]
    summary = {
        "manifest_items": len(manifest_ids),
        "raw_attempt_rows": len(attempts),
        "canonical_successes": len(rows),
        "missing_or_error_only": missing,
        "error_only_items": sorted(error_only),
        "unexpected_items": sorted(unexpected),
        "duplicate_attempt_rows": sum(max(0, value - 1) for value in attempt_counts.values()),
        "items_with_multiple_successes": sorted(
            item_id for item_id, value in successful_counts.items() if value > 1
        ),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows, summary = canonicalize(
        load_jsonl(args.manifest), load_jsonl(args.input)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary.update(
        {
            "input": str(args.input),
            "input_sha256": sha256(args.input),
            "output": str(args.out),
            "output_sha256": sha256(args.out),
        }
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
