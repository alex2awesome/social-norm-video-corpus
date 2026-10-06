#!/usr/bin/env python3
"""Evaluate a temporal verifier against a frozen manual development ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate(
    gold_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    minimum_precision: float = 0.90,
    minimum_recall: float = 0.50,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold: dict[int, bool] = {}
    for row in gold_rows:
        index = int(row["audit_index"])
        if index in gold:
            raise ValueError(f"duplicate gold index: {index}")
        if row["manual_gold"] not in {"positive", "negative"}:
            raise ValueError(f"invalid manual gold for {index}")
        gold[index] = row["manual_gold"] == "positive"

    outputs: dict[int, dict[str, Any]] = {}
    for row in output_rows:
        index = int(row["audit_index"])
        if index in outputs:
            raise ValueError(f"duplicate model output index: {index}")
        outputs[index] = row
    missing = sorted(set(gold) - set(outputs))
    extra = sorted(set(outputs) - set(gold))
    if missing or extra:
        raise ValueError(f"index mismatch: missing={missing}, extra={extra}")

    tp = fp = tn = fn = errors = 0
    disagreements: list[dict[str, Any]] = []
    for index, expected in gold.items():
        row = outputs[index]
        errored = bool(row.get("error"))
        predicted = bool(row.get("strict_pass")) and not errored
        errors += int(errored)
        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif not expected and not predicted:
            tn += 1
        else:
            fn += 1
        if predicted != expected or errored:
            disagreements.append(
                {
                    "audit_index": index,
                    "error": row.get("error"),
                    "expected": "positive" if expected else "negative",
                    "predicted": "positive" if predicted else "negative",
                    "stage_a": row.get("stage_a"),
                    "stage_b": row.get("stage_b"),
                }
            )

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    candidate_for_holdout = (
        errors == 0
        and precision is not None
        and recall is not None
        and precision >= minimum_precision
        and recall >= minimum_recall
    )
    summary = {
        "balanced_accuracy": balanced_accuracy,
        "candidate_for_disjoint_holdout": candidate_for_holdout,
        "coverage": len(output_rows) / len(gold_rows) if gold_rows else 0.0,
        "errors": errors,
        "false_negative_indices": [
            row["audit_index"]
            for row in disagreements
            if row["expected"] == "positive" and row["predicted"] == "negative"
        ],
        "false_positive_indices": [
            row["audit_index"]
            for row in disagreements
            if row["expected"] == "negative" and row["predicted"] == "positive"
        ],
        "fn": fn,
        "fp": fp,
        "minimum_precision": minimum_precision,
        "minimum_recall": minimum_recall,
        "n": len(gold),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tn": tn,
        "tp": tp,
    }
    return summary, disagreements


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--disagreements", required=True, type=Path)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--minimum-recall", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, disagreements = evaluate(
        read_jsonl(args.gold),
        read_jsonl(args.outputs),
        minimum_precision=args.minimum_precision,
        minimum_recall=args.minimum_recall,
    )
    if args.summary.exists():
        raise FileExistsError(f"refusing to overwrite: {args.summary}")
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_jsonl(args.disagreements, disagreements)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
