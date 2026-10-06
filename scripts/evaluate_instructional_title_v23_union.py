#!/usr/bin/env python3
"""Evaluate complementary instructional title and V23 demo signals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def metric(gold: dict[str, bool], prediction: dict[str, bool]) -> dict[str, Any]:
    tp = sum(gold[item] and prediction[item] for item in gold)
    fp = sum(not gold[item] and prediction[item] for item in gold)
    fn = sum(gold[item] and not prediction[item] for item in gold)
    tn = sum(not gold[item] and not prediction[item] for item in gold)
    return {
        "items": len(gold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "selected": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision_wilson_95": wilson(tp, tp + fp),
        "recall_wilson_95": wilson(tp, tp + fn),
    }


def evaluate(
    title_rows: list[dict[str, Any]],
    vlm_rows: list[dict[str, Any]],
    cohort: str,
) -> dict[str, Any]:
    titles = {
        str(row["item_id"]): row
        for row in title_rows
        if row.get("cohort") == cohort
    }
    vlm = {str(row["item_id"]): row for row in vlm_rows}
    if len(titles) != len([row for row in title_rows if row.get("cohort") == cohort]):
        raise ValueError("duplicate title item_id")
    if len(vlm) != len(vlm_rows):
        raise ValueError("duplicate VLM item_id")
    if set(titles) != set(vlm):
        raise ValueError("title and VLM cohorts differ")
    gold = {item: bool(vlm[item]["manual_usable_demo"]) for item in vlm}
    title = {item: bool(titles[item]["scene_title_candidate"]) for item in vlm}
    consensus = {item: bool(vlm[item]["pipeline_consensus"]) for item in vlm}
    predictions = {
        "title_cue": title,
        "v23_consensus": consensus,
        "title_or_v23": {item: title[item] or consensus[item] for item in vlm},
        "title_and_v23": {item: title[item] and consensus[item] for item in vlm},
    }
    disagreements = [
        {
            "item_id": item,
            "title_cue": title[item],
            "v23_consensus": consensus[item],
            "manual_usable_demo": gold[item],
        }
        for item in sorted(vlm)
        if title[item] != consensus[item]
    ]
    return {
        "kind": "instructional_title_v23_union_discovery_v1",
        "cohort": cohort,
        "items": len(vlm),
        "manual_usable_demos": sum(gold.values()),
        "rules": {name: metric(gold, values) for name, values in predictions.items()},
        "signal_overlap_selected": sum(title[item] and consensus[item] for item in vlm),
        "disagreements": disagreements,
        "status": "post_hoc_discovery_requires_fresh_source_disjoint_validation",
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title-records", type=Path, required=True)
    parser.add_argument("--vlm-evaluated", type=Path, required=True)
    parser.add_argument("--cohort", default="v22_fresh_source_disjoint")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    report = evaluate(
        read_jsonl(args.title_records), read_jsonl(args.vlm_evaluated), args.cohort
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
