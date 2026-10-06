#!/usr/bin/env python3
"""Merge resumable base/shard shadow scores into one verified canonical ledger."""

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def successful(row: dict[str, Any]) -> bool:
    return row.get("error") is None and bool(row.get("low_level"))


def merge_scores(
    manifest_rows: list[dict[str, Any]],
    score_groups: list[list[dict[str, Any]]],
    *,
    require_complete: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_ids = [row.get("item_id") for row in manifest_rows]
    if any(not item_id for item_id in manifest_ids):
        raise ValueError("manifest contains a missing item_id")
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("manifest item_id values are not unique")
    allowed = set(manifest_ids)
    selected: dict[str, dict[str, Any]] = {}
    attempts = 0
    unknown = set()
    duplicate_attempts = 0
    for rows in score_groups:
        for row in rows:
            item_id = row.get("item_id")
            if not item_id:
                continue
            attempts += 1
            if item_id not in allowed:
                unknown.add(str(item_id))
                continue
            previous = selected.get(item_id)
            if previous is not None:
                duplicate_attempts += 1
            if previous is None or successful(row) or not successful(previous):
                selected[item_id] = row
    missing = [item_id for item_id in manifest_ids if item_id not in selected]
    if unknown:
        raise ValueError(f"score files contain {len(unknown)} unknown item ids")
    if require_complete and missing:
        raise ValueError(f"missing score attempts for {len(missing)} manifest items")
    merged = [selected[item_id] for item_id in manifest_ids if item_id in selected]
    succeeded = sum(successful(row) for row in merged)
    summary = {
        "manifest_items": len(manifest_ids),
        "input_attempt_records": attempts,
        "duplicate_attempt_records": duplicate_attempts,
        "canonical_items": len(merged),
        "successful_items": succeeded,
        "failed_or_missing_media_items": len(merged) - succeeded,
        "unattempted_items": len(missing),
        "complete": not missing,
        "corpus_mutated": False,
    }
    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.out.exists() or args.summary.exists():
        raise SystemExit("refusing to overwrite output or summary")
    merged, summary = merge_scores(
        read_jsonl(args.manifest),
        [read_jsonl(path) for path in args.scores],
        require_complete=args.require_complete,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        for row in merged:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary.update(
        {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "score_inputs": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in args.scores
            ],
            "out": str(args.out.resolve()),
            "out_sha256": sha256(args.out),
        }
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
