#!/usr/bin/env python3
"""Validate and summarize a manual audit of commentary spatial repairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def evaluate(plan_path: Path, manifest_path: Path, ledger_path: Path) -> dict:
    plan = load_jsonl(plan_path)
    manifest = load_jsonl(manifest_path)
    with ledger_path.open(newline="") as handle:
        ledger = list(csv.DictReader(handle, delimiter="\t"))

    plan_variants = {str(row["variant_id"]) for row in plan}
    manifest_variants = {str(row["variant_id"]) for row in manifest}
    ledger_variants = {str(row["variant_id"]) for row in ledger}
    if not plan or len(plan_variants) != len(plan):
        raise ValueError("repair plan is empty or has duplicate variants")
    if plan_variants != manifest_variants or plan_variants != ledger_variants:
        raise ValueError("plan, rendered manifest, and ledger cover different variants")

    indices = [int(row["audit_index"]) for row in ledger]
    if len(set(indices)) != len(indices):
        raise ValueError("manual ledger has duplicate audit indices")
    required = {
        "outcome",
        "action_visible",
        "label_overlay_clean",
        "bounds_clean",
        "next_action",
        "manual_evidence",
    }
    for row in ledger:
        if not required.issubset(row) or not all(row[field].strip() for field in required):
            raise ValueError("manual ledger has an incomplete row")

    outcomes = Counter(row["outcome"] for row in ledger)
    visible = sum(row["action_visible"] == "yes" for row in ledger)
    label_clean = sum(row["label_overlay_clean"] == "yes" for row in ledger)
    ready_for_second_transform = sum(
        row["outcome"].startswith("repair_") for row in ledger
    )
    failed = sum(row["outcome"].startswith("fail_") for row in ledger)
    passed = sum(row["outcome"].startswith("pass_") for row in ledger)
    return {
        "kind": "commentary_spatial_repair_manual_audit",
        "policy": "shadow_only_no_corpus_mutation",
        "variants": len(ledger),
        "visible_action": visible,
        "label_overlay_clean": label_clean,
        "ready_for_second_transform": ready_for_second_transform,
        "failed_spatial_repair": failed,
        "passed_post_transform": passed,
        "outcome_counts": dict(sorted(outcomes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.plan, args.manifest, args.ledger)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
