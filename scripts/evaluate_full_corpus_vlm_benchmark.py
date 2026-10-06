#!/usr/bin/env python3
"""Evaluate frozen Qwen VLM outputs against every manually audited item."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_successes(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("error") is None and isinstance(row.get("result"), dict):
            result[str(row["item_id"])] = row
    return result


def v6_literal_event(result: dict[str, Any]) -> bool:
    return result.get("visually_observable_event") == "yes"


def v5_clean_demo(result: dict[str, Any]) -> bool:
    yes_fields = (
        "social_norm_domain",
        "behavior_occurs_in_scene",
        "affected_party_or_shared_setting_visible",
        "situated_interaction_complete",
        "usable_demo_after_relabel",
    )
    return (
        all(result.get(field) == "yes" for field in yes_fields)
        and result.get("audience_directed_explanation_only") == "no"
        and result.get("formal_procedure_trait_or_skill_only") == "no"
        and result.get("localization_quality") == "clean"
        and result.get("rejection_reason") == "none"
    )


def v5_exact_label(result: dict[str, Any]) -> bool:
    return v5_clean_demo(result) and result.get("proposed_norm_supported") == "yes"


def v7c_strict_witnessed(result: dict[str, Any]) -> bool:
    action_end = result.get("action_end_percent", -1)
    reaction_start = result.get("reaction_start_percent", -1)
    return (
        result.get("action_visible") == "yes"
        and result.get("action_voluntary") == "yes"
        and result.get("expectation_kind")
        in {"interpersonal_treatment", "shared_public_conduct"}
        and result.get("reaction_visible_or_audibly_grounded") == "yes"
        and result.get("reaction_source_role")
        in {"bystander", "authority_or_host", "organic_audience"}
        and result.get("reaction_content")
        in {
            "targeted_objection",
            "correction_or_sanction",
            "protective_intervention",
        }
        and result.get("action_established_before_reaction") == "yes"
        and result.get("reaction_targets_action") == "yes"
        and result.get("authenticity") in {"organic", "hidden_camera_genuine"}
        and result.get("proposed_label_relation") in {"exact", "repairable"}
        and result.get("pre_reaction_demo_quality")
        in {"clear_visual", "clear_audiovisual"}
        and isinstance(action_end, (int, float))
        and isinstance(reaction_start, (int, float))
        and 0 <= action_end <= reaction_start <= 100
    )


def binary_metrics(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    gold_field: str,
) -> dict[str, Any]:
    missing = [row["item_id"] for row in rows if row["item_id"] not in predictions]
    if missing:
        raise ValueError(f"missing predictions: {missing}")
    gold = np.asarray([bool(row[gold_field]) for row in rows], dtype=int)
    predicted = np.asarray(
        [predicate(predictions[row["item_id"]]["result"]) for row in rows],
        dtype=int,
    )
    tn, fp, fn, tp = confusion_matrix(gold, predicted, labels=[0, 1]).ravel()
    return {
        "items": len(rows),
        "gold_positive": int(gold.sum()),
        "selected": int(predicted.sum()),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "balanced_accuracy": float(balanced_accuracy_score(gold, predicted)),
        "selected_item_ids": [
            row["item_id"] for row, value in zip(rows, predicted) if value
        ],
        "false_positive_item_ids": [
            row["item_id"]
            for row, predicted_value, gold_value in zip(rows, predicted, gold)
            if predicted_value and not gold_value
        ],
        "false_negative_item_ids": [
            row["item_id"]
            for row, predicted_value, gold_value in zip(rows, predicted, gold)
            if not predicted_value and gold_value
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--v6", required=True, type=Path)
    parser.add_argument("--v5", required=True, type=Path)
    parser.add_argument("--v7c", required=True, type=Path)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit item-id lists while retaining every aggregate metric.",
    )
    args = parser.parse_args()
    manifest = read_jsonl(args.manifest)
    v6 = latest_successes(args.v6)
    v5 = latest_successes(args.v5)
    v7c = latest_successes(args.v7c)
    report: dict[str, Any] = {
        "policy": "evaluate_all_frozen_human_labels_no_rule_promotion",
        "manifest_items": len(manifest),
        "successful_predictions": {
            "v6": len(v6),
            "v5": len(v5),
            "v7c_witnessed": len(v7c),
        },
        "cohorts": {},
    }
    for cohort in ("mechanism", "uniform"):
        cohort_result: dict[str, Any] = {}
        for pillar in ("instructional", "witnessed", "commentary"):
            rows = [
                row
                for row in manifest
                if row["cohort"] == cohort and row["pillar"] == pillar
            ]
            pillar_result = {
                "v6_literal_event_vs_visual": binary_metrics(
                    rows, v6, v6_literal_event, "gold_scene_visible"
                ),
                "v5_clean_demo_vs_visual": binary_metrics(
                    rows, v5, v5_clean_demo, "gold_scene_visible"
                ),
                "v5_exact_label_vs_strict": binary_metrics(
                    rows, v5, v5_exact_label, "gold_usable"
                ),
            }
            if pillar == "witnessed":
                pillar_result["v7c_strict_vs_witnessed_strict"] = binary_metrics(
                    rows, v7c, v7c_strict_witnessed, "gold_usable"
                )
            cohort_result[pillar] = pillar_result
        report["cohorts"][cohort] = cohort_result
    if args.compact:
        for cohort in report["cohorts"].values():
            for pillar in cohort.values():
                for evaluation in pillar.values():
                    for key in list(evaluation):
                        if key.endswith("_item_ids"):
                            del evaluation[key]
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
