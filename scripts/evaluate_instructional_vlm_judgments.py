#!/usr/bin/env python3
"""Compare one append-only VLM judgment stream with frozen manual labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_tsv(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {int(row["audit_index"]): row for row in rows}


def latest_success(
    rows: list[dict[str, Any]], model: str | None
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if model is not None and row.get("model") != model:
            continue
        if row.get("error") is None and isinstance(row.get("result"), dict):
            latest[row["item_id"]] = row
    return latest


def nested(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def rates(tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    total = tp + fp + tn + fn
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "accuracy": (tp + tn) / total if total else None,
    }


def evaluate(
    selection_path: Path,
    visual_path: Path,
    semantic_path: Path,
    predictions_path: Path,
    prediction_field: str,
    target: str,
    model: str | None = None,
    embedded: bool = False,
) -> dict[str, Any]:
    selection_rows = load_jsonl(selection_path)
    visual = load_tsv(visual_path)
    semantic = load_tsv(semantic_path)
    expected = set(range(len(selection_rows)))
    if set(visual) != expected or set(semantic) != expected:
        raise ValueError("manual ledgers do not exactly cover selection")
    predictions = (
        {row["item_id"]: row for row in selection_rows}
        if embedded
        else latest_success(load_jsonl(predictions_path), model)
    )
    if target not in {"visual", "exact"}:
        raise ValueError("target must be visual or exact")

    tp = fp = tn = fn = 0
    missing: list[int] = []
    uncertain: list[int] = []
    false_positive: list[int] = []
    false_negative: list[int] = []
    for source in selection_rows:
        index = int(source["audit_index"])
        truth = (
            visual[index]["visual_demo"] == "Y"
            if target == "visual"
            else semantic[index]["exact_original"] == "Y"
        )
        record = predictions.get(source["item_id"])
        raw = nested(record, prediction_field) if record else None
        if raw is None:
            missing.append(index)
        elif raw not in {"yes", "no"}:
            uncertain.append(index)
        predicted = raw == "yes"
        if predicted and truth:
            tp += 1
        elif predicted and not truth:
            fp += 1
            false_positive.append(index)
        elif not predicted and truth:
            fn += 1
            false_negative.append(index)
        else:
            tn += 1
    return {
        "kind": "instructional_vlm_manual_comparison",
        "target": target,
        "prediction_field": prediction_field,
        "model": model,
        "embedded_predictions": embedded,
        "audited_items": len(selection_rows),
        "successful_prediction_records": len(predictions),
        "missing_indices": missing,
        "uncertain_indices": uncertain,
        "fail_closed_confusion": rates(tp, fp, tn, fn),
        "false_positive_indices": false_positive,
        "false_negative_indices": false_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--visual-ledger", type=Path, required=True)
    parser.add_argument("--semantic-ledger", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-field", required=True)
    parser.add_argument("--target", choices=("visual", "exact"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--embedded", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = evaluate(
        args.selection,
        args.visual_ledger,
        args.semantic_ledger,
        args.predictions,
        args.prediction_field,
        args.target,
        args.model,
        args.embedded,
    )
    args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
