#!/usr/bin/env python3
"""Select a recoverable instructional rescore queue from existing shadow data.

The selector is intentionally a recall-oriented queue, not a corpus decision.
It uses the already-computed Qwen v4 scene vote and independent text gate to
avoid moving all 42k proxies between servers before the stricter v5 pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def latest_successes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    successful: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("result") is not None and not row.get("error"):
            successful[row["item_id"]] = row
    return successful


def first_stage_pass(
    qwen_result: dict[str, Any], text_result: dict[str, Any]
) -> bool:
    return (
        qwen_result.get("social_norm_domain") == "yes"
        and qwen_result.get("usable_demo_after_relabel") == "yes"
        and qwen_result.get("localization_quality") == "clean"
        and text_result.get("social_norm_candidate") == "yes"
    )


def select(
    manifest_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qwen = latest_successes(qwen_rows)
    text = latest_successes(text_rows)
    selected: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    polarities: Counter[str] = Counter()
    uids: set[str] = set()
    for row in manifest_rows:
        item_id = row["item_id"]
        if item_id not in qwen or item_id not in text:
            continue
        if not first_stage_pass(
            qwen[item_id]["result"], text[item_id]["result"]
        ):
            continue
        record = dict(row)
        record["first_stage_qwen_v4_result"] = qwen[item_id]["result"]
        record["first_stage_text_v2_result"] = text[item_id]["result"]
        selected.append(record)
        categories[str(row.get("category") or "unknown")] += 1
        polarities[str(row.get("polarity") or "unknown")] += 1
        uids.add(str(row["uid"]))

    summary = {
        "kind": "instructional_v5_rescore_queue",
        "corpus_action": "none_shadow_only",
        "manifest_items": len(manifest_rows),
        "qwen_v4_valid": len(qwen),
        "text_v2_valid": len(text),
        "selected_items": len(selected),
        "selected_unique_uids": len(uids),
        "selected_fraction": (
            len(selected) / len(manifest_rows) if manifest_rows else 0.0
        ),
        "selection_rule": {
            "qwen_v4_social_norm_domain": "yes",
            "qwen_v4_usable_demo_after_relabel": "yes",
            "qwen_v4_localization_quality": "clean",
            "text_v2_social_norm_candidate": "yes",
        },
        "by_category": dict(sorted(categories.items())),
        "by_polarity": dict(sorted(polarities.items())),
    }
    return selected, summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen queue: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qwen-v4", type=Path, required=True)
    parser.add_argument("--text-v2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    selected, summary = select(
        load_jsonl(args.manifest),
        load_jsonl(args.qwen_v4),
        load_jsonl(args.text_v2),
    )
    write_jsonl(args.out, selected)
    summary["queue_sha256"] = sha256_file(args.out)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
