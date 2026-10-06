#!/usr/bin/env python3
"""Evaluate instructional polarity solely as a manual-review ordering signal.

This rule must never reject or accept a demo.  It asks whether clips whose LLM
polarity is not ``explanation`` should be reviewed before explanation clips,
while preserving every clip for the independent visual-demo contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


NON_EXPLANATION = {"violation", "correct", "contrast"}
GATE = {
    "minimum_items_per_cohort": 100,
    "minimum_selected_per_cohort": 70,
    "minimum_visual_demo_recall_per_cohort": 0.85,
    "maximum_explanation_visual_demo_rate_per_cohort": 0.15,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if not n:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = p + z * z / (2 * n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n**2))
    return [(center - radius) / denominator, (center + radius) / denominator]


def fresh_rows(
    selection: list[dict[str, Any]], ledger: list[dict[str, str]]
) -> list[dict[str, Any]]:
    manual = {int(row["audit_index"]): row for row in ledger}
    if len(manual) != len(ledger):
        raise ValueError("duplicate fresh audit index")
    expected = {int(row["audit_index"]) for row in selection}
    if set(manual) != expected:
        raise ValueError("fresh manual ledger does not exactly cover selection")
    uids = [str(row["uid"]) for row in selection]
    if len(uids) != len(set(uids)):
        raise ValueError("fresh cohort is not source-disjoint")
    return [
        {
            "uid": str(row["uid"]),
            "polarity": str(row.get("polarity") or "unknown"),
            "visual_demo": manual[int(row["audit_index"])]["visual_demo"] == "yes",
        }
        for row in selection
    ]


def replication_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uids = [str(row["uid"]) for row in rows]
    if len(uids) != len(set(uids)):
        raise ValueError("replication cohort is not source-disjoint")
    return [
        {
            "uid": str(row["uid"]),
            "polarity": str(row.get("polarity") or "unknown"),
            "visual_demo": bool(row["manual_visual_demo"]),
        }
        for row in rows
    ]


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["polarity"] in NON_EXPLANATION]
    tp = sum(row["visual_demo"] for row in selected)
    fp = len(selected) - tp
    excluded = [row for row in rows if row["polarity"] not in NON_EXPLANATION]
    fn = sum(row["visual_demo"] for row in excluded)
    tn = len(excluded) - fn
    explanation = [row for row in rows if row["polarity"] == "explanation"]
    explanation_visual = sum(row["visual_demo"] for row in explanation)
    positives = tp + fn
    return {
        "items": len(rows),
        "selected": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / len(selected) if selected else None,
        "precision_wilson_95": wilson(tp, len(selected)),
        "recall": tp / positives if positives else None,
        "recall_wilson_95": wilson(tp, positives),
        "explanation_items": len(explanation),
        "explanation_visual_demos": explanation_visual,
        "explanation_visual_demo_rate": (
            explanation_visual / len(explanation) if explanation else None
        ),
        "by_polarity": {
            polarity: {
                "items": len(group),
                "visual_demos": sum(row["visual_demo"] for row in group),
                "visual_demo_rate": sum(row["visual_demo"] for row in group) / len(group),
            }
            for polarity in sorted({row["polarity"] for row in rows})
            if (group := [row for row in rows if row["polarity"] == polarity])
        },
    }


def evaluate(cohorts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    metrics = {name: metric(rows) for name, rows in cohorts.items()}
    checks: dict[str, bool] = {}
    for name, value in metrics.items():
        checks[f"{name}_minimum_items"] = value["items"] >= GATE["minimum_items_per_cohort"]
        checks[f"{name}_minimum_selected"] = value["selected"] >= GATE["minimum_selected_per_cohort"]
        checks[f"{name}_minimum_recall"] = value["recall"] is not None and value["recall"] >= GATE["minimum_visual_demo_recall_per_cohort"]
        checks[f"{name}_maximum_explanation_rate"] = (
            value["explanation_visual_demo_rate"] is not None
            and value["explanation_visual_demo_rate"] <= GATE["maximum_explanation_visual_demo_rate_per_cohort"]
        )
    passed = all(checks.values())
    all_rows = [row for rows in cohorts.values() for row in rows]
    return {
        "kind": "instructional_polarity_review_priority_v1",
        "cohorts": metrics,
        "combined": metric(all_rows),
        "source_disjoint_within_each_cohort": True,
        "gate": GATE,
        "gate_checks": checks,
        "review_ranking_promoted": passed,
        "allowed_use": "manual_review_ranking_only" if passed else "none",
        "trigger_polarities": sorted(NON_EXPLANATION) if passed else [],
        "explanation_policy": "preserve_and_review_at_lower_priority_never_reject",
        "automatic_acceptance": False,
        "automatic_rejection": False,
        "corpus_mutated": False,
        "cohort_sizes": dict(sorted(Counter(name for name, rows in cohorts.items() for _ in rows).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-selection", type=Path, required=True)
    parser.add_argument("--fresh-ledger", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    report = evaluate({
        "fresh_100": fresh_rows(read_jsonl(args.fresh_selection), read_tsv(args.fresh_ledger)),
        "replication_100": replication_rows(read_jsonl(args.replication)),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["review_ranking_promoted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
