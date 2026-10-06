#!/usr/bin/env python3
"""Evaluate a frozen commentary VLM conjunction on old and fresh audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def latest_success(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        row["item_id"]: row["result"]
        for row in rows
        if row.get("error") is None and isinstance(row.get("result"), dict)
    }


def predict(result: dict[str, Any]) -> bool:
    return (
        result.get("behavior_occurs_in_scene") == "yes"
        and result.get("social_norm_domain") == "yes"
        and result.get("proposed_norm_supported") == "yes"
        and result.get("depiction_type") == "organic_scene"
    )


def evaluate(gold_rows: list[dict[str, Any]], vlm_rows: list[dict[str, Any]]) -> dict:
    gold = {
        row["item_id"]: row["strict_commentary_visual_pass"] == "yes"
        for row in gold_rows
    }
    vlm = latest_success(vlm_rows)
    if set(gold) - set(vlm):
        raise ValueError("VLM output does not cover every gold item")
    selected = {item_id for item_id in gold if predict(vlm[item_id])}
    tp = sorted(item_id for item_id in selected if gold[item_id])
    fp = sorted(item_id for item_id in selected if not gold[item_id])
    fn = sorted(item_id for item_id in gold if gold[item_id] and item_id not in selected)
    return {
        "items": len(gold),
        "gold_positives": sum(gold.values()),
        "selected": len(selected),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": len(tp) / len(selected) if selected else None,
        "recall": len(tp) / (len(tp) + len(fn)) if tp or fn else None,
        "true_positive_item_ids": tp,
        "false_positive_item_ids": fp,
        "false_negative_item_ids": fn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-gold", type=Path, action="append", required=True)
    parser.add_argument("--old-vlm", type=Path, required=True)
    parser.add_argument("--fresh-gold", type=Path, required=True)
    parser.add_argument("--fresh-vlm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    old_gold = [
        row for path in args.old_gold for row in read_jsonl(path)
    ]
    fresh_gold = read_jsonl(args.fresh_gold)
    if not fresh_gold:
        raise ValueError("fresh gold is empty")
    report = {
        "kind": "commentary_vlm_conjunction_replication",
        "frozen_rule": (
            "behavior_occurs=yes AND social_norm_domain=yes AND "
            "proposed_norm_supported=yes AND depiction_type=organic_scene"
        ),
        "fresh_12": evaluate(fresh_gold, read_jsonl(args.fresh_vlm)),
        "prior_100": evaluate(old_gold, read_jsonl(args.old_vlm)),
        "decision": "reject_no_generalization",
        "policy": "shadow_only_no_keep_reject",
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
