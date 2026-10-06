#!/usr/bin/env python3
"""Evaluate fallible VLM predictions against a complete manual canary audit."""

from __future__ import annotations

import argparse
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


def positive_score(label: str, confidence: float) -> float:
    confidence = min(1.0, max(0.0, float(confidence)))
    if label == "yes":
        return confidence
    if label == "no":
        return 1.0 - confidence
    if label == "uncertain":
        return 0.5
    raise ValueError(f"unsupported prediction label: {label!r}")


def gold_labels(review: dict[str, Any]) -> dict[str, bool]:
    visual_social_demo = (
        review["visual_demo_present"] == "yes"
        and review["is_social_norm"] == "yes"
    )
    return {
        "keep_after_relabel": visual_social_demo,
        "strict_current_label": (
            visual_social_demo and review["norm_supported"] == "yes"
        ),
    }


def confusion(
    rows: list[dict[str, Any]],
    *,
    gold_key: str,
    prediction_key: str,
    threshold: float,
) -> dict[str, Any]:
    counts = Counter()
    disagreements = []
    for row in rows:
        result = row["prediction"]["result"]
        predicted = (
            positive_score(result[prediction_key], result["confidence"])
            >= threshold
        )
        gold = row["gold"][gold_key]
        outcome = (
            "tp" if predicted and gold
            else "fp" if predicted
            else "fn" if gold
            else "tn"
        )
        counts[outcome] += 1
        if outcome in {"fp", "fn"}:
            disagreements.append(
                {
                    "item_id": row["item_id"],
                    "ordinal": row["review"]["ordinal"],
                    "outcome": outcome,
                    "manual_decision": row["review"]["decision"],
                    "manual_evidence": row["review"]["evidence"],
                    "prediction_label": result[prediction_key],
                    "prediction_confidence": result["confidence"],
                    "prediction_evidence": result["evidence"],
                }
            )
    tp, fp, tn, fn = (
        counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    )
    return {
        "n": len(rows),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "disagreements": disagreements,
    }


def build_report(
    manifest: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    prediction_keys: list[str],
    require_complete: bool,
) -> dict[str, Any]:
    manifest_by_id = {row["item_id"]: row for row in manifest}
    reviews_by_id = {row["item_id"]: row for row in reviews}
    if len(manifest_by_id) != len(manifest):
        raise ValueError("manifest contains duplicate item_id values")
    if len(reviews_by_id) != len(reviews):
        raise ValueError("manual review contains duplicate item_id values")
    if set(reviews_by_id) != set(manifest_by_id):
        raise ValueError("manual review must cover the manifest exactly")
    for item_id, review in reviews_by_id.items():
        manifest_row = manifest_by_id[item_id]
        if (
            review["ordinal"],
            review["uid"],
        ) != (
            manifest_row["ordinal"],
            manifest_row["uid"],
        ):
            raise ValueError(f"manual identity mismatch: {item_id}")

    successful = [
        row
        for row in predictions
        if row.get("error") is None and row.get("result") is not None
    ]
    prediction_by_id = {row["item_id"]: row for row in successful}
    if len(prediction_by_id) != len(successful):
        raise ValueError("successful predictions contain duplicate item_id values")
    unknown = set(prediction_by_id) - set(manifest_by_id)
    if unknown:
        raise ValueError(f"predictions outside manifest: {sorted(unknown)[:5]}")
    missing = sorted(set(manifest_by_id) - set(prediction_by_id))
    if require_complete and missing:
        raise ValueError(f"incomplete predictions: {len(missing)} missing")

    joined = [
        {
            "item_id": item_id,
            "manifest": manifest_by_id[item_id],
            "review": reviews_by_id[item_id],
            "gold": gold_labels(reviews_by_id[item_id]),
            "prediction": prediction_by_id[item_id],
        }
        for item_id in sorted(
            prediction_by_id, key=lambda value: manifest_by_id[value]["ordinal"]
        )
    ]
    tasks = {}
    for gold_key in ("strict_current_label", "keep_after_relabel"):
        tasks[gold_key] = {}
        for prediction_key in prediction_keys:
            tasks[gold_key][prediction_key] = [
                confusion(
                    joined,
                    gold_key=gold_key,
                    prediction_key=prediction_key,
                    threshold=threshold,
                )
                for threshold in (0.5, 0.6, 0.7, 0.8, 0.9)
            ]
    return {
        "manifest_items": len(manifest),
        "manual_reviews": len(reviews),
        "successful_predictions": len(joined),
        "failed_predictions": len(predictions) - len(successful),
        "missing_prediction_item_ids": missing,
        "manual_gold_counts": {
            key: sum(gold_labels(row)[key] for row in reviews)
            for key in ("strict_current_label", "keep_after_relabel")
        },
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--prediction-key",
        action="append",
        dest="prediction_keys",
        default=[],
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    prediction_keys = args.prediction_keys or [
        "target_pillar_signal_visible",
        "social_behavior_scene_visible",
    ]
    report = build_report(
        load_jsonl(args.manifest),
        load_jsonl(args.manual_review),
        load_jsonl(args.predictions),
        prediction_keys=prediction_keys,
        require_complete=args.require_complete,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manual_reviews": report["manual_reviews"],
                "successful_predictions": report["successful_predictions"],
                "missing_predictions": len(report["missing_prediction_item_ids"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
