#!/usr/bin/env python3
"""Evaluate commentary VLM shadow scores against the frozen manual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET_CANDIDATE_ROUTES = {
    "dense_followup_commentary",
    "dense_followup_instructional",
}
VISUAL_RECOVERY_ROUTES = TARGET_CANDIDATE_ROUTES | {
    "dense_recut_review",
    "dense_source_mining_review",
}
CLEAR_DEMO_QUALITIES = {"clear_visual", "clear_audiovisual"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_model_rows(base_rows: list[dict], override_rows: list[dict]) -> list[dict]:
    """Replace failed/base rows by item_id without mutating either input list."""
    merged = {row["item_id"]: row for row in base_rows}
    if len(merged) != len(base_rows):
        raise ValueError("base model output contains duplicate item_id values")
    for row in override_rows:
        merged[row["item_id"]] = row
    return list(merged.values())


def manual_positive(row: dict, *, strict_clear: bool) -> bool:
    if row["expected_disposition"] not in TARGET_CANDIDATE_ROUTES:
        return False
    return not strict_clear or row["demo_quality"] in CLEAR_DEMO_QUALITIES


def predict(row: dict, rubric: str) -> bool | None:
    if row.get("error") or not isinstance(row.get("result"), dict):
        return None
    result = row["result"]
    if rubric == "v6_blind":
        return (
            str(result.get("visually_observable_event", "")).lower() == "yes"
            and str(result.get("presentation_or_context_only", "")).lower() != "yes"
        )
    if rubric == "v3_conditioned":
        required = (
            "situated_social_scenario_visible",
            "concrete_action_or_situated_speech_visible",
            "proposed_norm_plausibly_demonstrated",
        )
        return all(str(result.get(key, "")).lower() == "yes" for key in required)
    raise ValueError(f"unsupported rubric: {rubric}")


def metrics(gold: dict[str, bool], predictions: dict[str, bool | None]) -> dict:
    evaluated = [item_id for item_id in gold if predictions.get(item_id) is not None]
    tp = sum(gold[item_id] and predictions[item_id] is True for item_id in evaluated)
    fp = sum(not gold[item_id] and predictions[item_id] is True for item_id in evaluated)
    fn = sum(gold[item_id] and predictions[item_id] is False for item_id in evaluated)
    tn = sum(not gold[item_id] and predictions[item_id] is False for item_id in evaluated)
    return {
        "gold_positive": sum(gold.values()),
        "evaluated": len(evaluated),
        "abstained_or_error": len(gold) - len(evaluated),
        "predicted_positive": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def evaluate_predictions(
    manual_rows: list[dict],
    predictions: dict[str, bool | None],
) -> dict:
    manual = {row["item_id"]: row for row in manual_rows}
    if len(manual) != len(manual_rows):
        raise ValueError("manual review contains duplicate item_id values")
    predictions = {
        item_id: predictions.get(item_id)
        for item_id in manual
    }
    reports = {}
    for name, gold_selector in (
        (
            "all_target_candidates",
            lambda row: manual_positive(row, strict_clear=False),
        ),
        (
            "strict_clear_targets",
            lambda row: manual_positive(row, strict_clear=True),
        ),
        (
            "visual_recovery_candidates",
            lambda row: row["expected_disposition"] in VISUAL_RECOVERY_ROUTES,
        ),
    ):
        gold = {
            item_id: gold_selector(row)
            for item_id, row in manual.items()
        }
        report = metrics(gold, predictions)
        report["predicted_positive_items"] = [
            item_id for item_id in manual if predictions[item_id] is True
        ]
        report["true_positive_items"] = [
            item_id
            for item_id in manual
            if predictions[item_id] is True and gold[item_id]
        ]
        report["false_positive_items"] = [
            item_id
            for item_id in manual
            if predictions[item_id] is True and not gold[item_id]
        ]
        report["false_negative_items"] = [
            item_id
            for item_id in manual
            if predictions[item_id] is False and gold[item_id]
        ]
        report["true_negative_items"] = [
            item_id
            for item_id in manual
            if predictions[item_id] is False and not gold[item_id]
        ]
        reports[name] = report
    return reports


def model_predictions(
    manual_rows: list[dict],
    model_rows: list[dict],
    *,
    rubric: str,
) -> dict[str, bool | None]:
    manual_ids = {row["item_id"] for row in manual_rows}
    model = {row["item_id"]: row for row in model_rows}
    if len(model) != len(model_rows):
        raise ValueError("model output contains duplicate item_id values")
    return {
        item_id: predict(model[item_id], rubric) if item_id in model else None
        for item_id in manual_ids
    }


def evaluate(
    manual_rows: list[dict],
    model_rows: list[dict],
    *,
    rubric: str,
) -> dict:
    predictions = model_predictions(manual_rows, model_rows, rubric=rubric)
    return {
        "status": "shadow_only_do_not_promote_without_manual_replication",
        "rubric": rubric,
        "manual_items": len(manual_rows),
        "model_items": len(model_rows),
        "metrics": evaluate_predictions(manual_rows, predictions),
    }


def evaluate_ensemble(
    manual_rows: list[dict],
    primary_rows: list[dict],
    secondary_rows: list[dict],
    *,
    rubric: str,
    operator: str,
) -> dict:
    primary = model_predictions(manual_rows, primary_rows, rubric=rubric)
    secondary = model_predictions(manual_rows, secondary_rows, rubric=rubric)
    predictions: dict[str, bool | None] = {}
    for item_id in primary:
        first, second = primary[item_id], secondary[item_id]
        if first is None or second is None:
            predictions[item_id] = None
        elif operator == "intersection":
            predictions[item_id] = first and second
        elif operator == "union":
            predictions[item_id] = first or second
        else:
            raise ValueError(f"unsupported ensemble operator: {operator}")
    return {
        "status": "shadow_only_do_not_promote_without_manual_replication",
        "rubric": rubric,
        "ensemble_operator": operator,
        "manual_items": len(manual_rows),
        "primary_model_items": len(primary_rows),
        "secondary_model_items": len(secondary_rows),
        "metrics": evaluate_predictions(manual_rows, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-override",
        type=Path,
        action="append",
        default=[],
        help="Optional retry JSONL whose item_id rows replace rows in --model.",
    )
    parser.add_argument("--secondary-model", type=Path)
    parser.add_argument(
        "--secondary-model-override",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--ensemble-operator",
        choices=("intersection", "union"),
        default="intersection",
    )
    parser.add_argument(
        "--rubric", choices=("v6_blind", "v3_conditioned"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model_rows = load_jsonl(args.model)
    for override_path in args.model_override:
        model_rows = merge_model_rows(model_rows, load_jsonl(override_path))
    manual_rows = load_jsonl(args.manual)
    if args.secondary_model:
        secondary_rows = load_jsonl(args.secondary_model)
        for override_path in args.secondary_model_override:
            secondary_rows = merge_model_rows(
                secondary_rows, load_jsonl(override_path)
            )
        report = evaluate_ensemble(
            manual_rows,
            model_rows,
            secondary_rows,
            rubric=args.rubric,
            operator=args.ensemble_operator,
        )
    else:
        report = evaluate(manual_rows, model_rows, rubric=args.rubric)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
