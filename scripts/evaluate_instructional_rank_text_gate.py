#!/usr/bin/env python3
"""Evaluate text semantics alone and after a visual-rank gate on manual gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def confusion(labels: list[bool], predictions: list[bool]) -> dict[str, Any]:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    tn = sum(not label and not prediction for label, prediction in zip(labels, predictions))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "selected": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def evaluate(
    sealed: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gold = {row["item_id"]: row for row in gold_rows}
    text = {
        row["item_id"]: row["result"]
        for row in text_rows
        if row.get("result") is not None and row.get("error") is None
    }
    expected = [row["item_id"] for row in sealed]
    if set(expected) != set(gold):
        raise ValueError("sealed/manual-gold coverage mismatch")
    missing = [item_id for item_id in expected if item_id not in text]
    joined = [row for row in sealed if row["item_id"] in text]

    def standard(result: dict[str, Any]) -> bool:
        return result["social_norm_candidate"] == "yes"

    def strict(result: dict[str, Any]) -> bool:
        return (
            standard(result)
            and result.get("concrete_behavior_named") == "yes"
            and result.get("affected_other_or_shared_setting_named") == "yes"
            and result.get("norm_type") in {"tacit_interpersonal", "shared_public"}
        )

    social_labels = [
        gold[row["item_id"]]["assigned_norm_is_social_norm"] == "yes" for row in joined
    ]
    strict_labels = [
        gold[row["item_id"]]["strict_instructional_pass"] == "yes" for row in joined
    ]
    standard_predictions = [standard(text[row["item_id"]]) for row in joined]
    strict_predictions = [strict(text[row["item_id"]]) for row in joined]
    top_predictions = [row["score_band"] == "top_quintile" for row in joined]
    combined_predictions = [
        top and semantic for top, semantic in zip(top_predictions, strict_predictions)
    ]
    selected_records = [
        {
            "item_id": row["item_id"],
            "score_band": row["score_band"],
            "gold_social": social,
            "gold_strict": strict_gold,
            "text_standard": standard_prediction,
            "text_strict": strict_prediction,
            "combined_top_and_text_strict": combined,
            "norm_type": text[row["item_id"]].get("norm_type"),
            "reason": text[row["item_id"]].get("reason"),
        }
        for (
            row,
            social,
            strict_gold,
            standard_prediction,
            strict_prediction,
            combined,
        ) in zip(
            joined,
            social_labels,
            strict_labels,
            standard_predictions,
            strict_predictions,
            combined_predictions,
        )
        if combined
    ]
    return {
        "kind": "instructional_rank_plus_text_gate_confirmation",
        "items_expected": len(expected),
        "items_joined": len(joined),
        "missing_text_scores": missing,
        "text_standard_vs_assigned_social": confusion(
            social_labels, standard_predictions
        ),
        "text_strict_vs_assigned_social": confusion(social_labels, strict_predictions),
        "visual_top_band_vs_strict_pass": confusion(strict_labels, top_predictions),
        "combined_top_and_text_strict_vs_strict_pass": confusion(
            strict_labels, combined_predictions
        ),
        "combined_selected_records": selected_records,
        "decision": {
            "automatic_keep": False,
            "policy": "benchmark_only_until_source_disjoint replication",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--text-scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    report = evaluate(
        read_jsonl(args.sealed),
        read_jsonl(args.gold),
        read_jsonl(args.text_scores),
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
