#!/usr/bin/env python3
"""Fit the audited low-level visual ranker and score the V20 population.

The fitted threshold is a development-band boundary only. Outputs are
append-only ranking evidence and never authorize keep/reject decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.benchmark_instructional_storyboard_clip import (
        metrics,
        oof_logistic,
        sha256,
    )
    from scripts.benchmark_instructional_v19_cheap_features import (
        best_precision_threshold,
        read_jsonl,
    )
except ModuleNotFoundError:
    from benchmark_instructional_storyboard_clip import (
        metrics,
        oof_logistic,
        sha256,
    )
    from benchmark_instructional_v19_cheap_features import (
        best_precision_threshold,
        read_jsonl,
    )


LOW_LEVEL = (
    "face_count_mean",
    "face_present_fraction",
    "hard_cut_fraction",
    "histogram_delta_mean",
    "motion_max",
    "motion_mean",
    "multiple_faces_fraction",
)


def require_fresh_outputs(output: Path, summary: Path) -> None:
    """Fail closed rather than replacing a frozen shadow expansion."""
    existing = [str(path) for path in (output, summary) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite frozen outputs: {existing}")


def last_successful_rows(
    paths: list[Path],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get("error") is None and row.get("item_id"):
                output[str(row["item_id"])] = row
    return output


def vector(row: dict[str, Any]) -> list[float]:
    low = row.get("low_level") or {}
    missing = [name for name in LOW_LEVEL if name not in low]
    if missing:
        raise ValueError(f"missing low-level features: {missing}")
    return [float(low[name]) for name in LOW_LEVEL]


def fit_ranker(
    manifest: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    minimum_predictions: int,
) -> tuple[Any, dict[str, Any]]:
    train = np.asarray(
        [row["audit_cohort"] in {"v15", "v17"} for row in manifest]
    )
    values = np.asarray(
        [vector(scores[str(row["item_id"])]) for row in manifest],
        dtype=np.float64,
    )
    labels = np.asarray(
        [bool(row["gold_scene_visible"]) for row in manifest]
    )
    groups = np.asarray([str(row["uid"]) for row in manifest])
    choices = []
    for c_value in (0.001, 0.01, 0.1, 1.0, 10.0):
        oof = oof_logistic(
            values[train], labels[train], groups[train], c_value
        )
        choices.append(
            (
                average_precision_score(labels[train], oof),
                c_value,
                oof,
            )
        )
    train_ap, c_value, oof = max(choices, key=lambda item: item[0])
    threshold, train_metrics = best_precision_threshold(
        labels[train], oof, minimum_predictions
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        ),
    )
    model.fit(values[train], labels[train])
    v18_scores = model.predict_proba(values[~train])[:, 1]
    return model, {
        "selected_c": c_value,
        "train_oof_average_precision": train_ap,
        "minimum_train_predictions": minimum_predictions,
        "threshold_fit_v15_v17_oof": threshold,
        "v15_v17_oof_metrics": train_metrics,
        "v18_manual_transfer_metrics": metrics(
            labels[~train], v18_scores >= threshold
        ),
        "feature_names": list(LOW_LEVEL),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument(
        "--benchmark-scores", type=Path, action="append", required=True
    )
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument(
        "--population-scores", type=Path, action="append", required=True
    )
    parser.add_argument("--minimum-train-predictions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    require_fresh_outputs(args.output, args.summary)

    benchmark_manifest = read_jsonl(args.benchmark_manifest)
    benchmark_scores = last_successful_rows(args.benchmark_scores)
    missing_benchmark = {
        str(row["item_id"]) for row in benchmark_manifest
    } - set(benchmark_scores)
    if missing_benchmark:
        raise ValueError(
            f"missing {len(missing_benchmark)} benchmark score rows"
        )
    model, fit = fit_ranker(
        benchmark_manifest,
        benchmark_scores,
        args.minimum_train_predictions,
    )

    population = read_jsonl(args.population_manifest)
    population_scores = last_successful_rows(args.population_scores)
    available = [
        row for row in population
        if str(row["item_id"]) in population_scores
    ]
    missing_population = len(population) - len(available)
    values = np.asarray(
        [
            vector(population_scores[str(row["item_id"])])
            for row in available
        ],
        dtype=np.float64,
    )
    probabilities = model.predict_proba(values)[:, 1]
    threshold = float(fit["threshold_fit_v15_v17_oof"])
    output_rows = [
        {
            **row,
            "visual_rank_score": float(score),
            "development_review_band": bool(score >= threshold),
            "ranker_protocol": "fit_v15_v17_visual_test_v18_manual",
            "ranker_policy": "review_priority_only_no_keep_reject",
        }
        for row, score in zip(available, probabilities)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows)
    )
    summary = {
        "kind": "instructional_v20_low_level_visual_shadow_ranking",
        "status": "posthoc_development_not_promotion",
        "policy": "review_priority_only_no_keep_reject_or_corpus_mutation",
        "fit": fit,
        "population": {
            "manifest_items": len(population),
            "successfully_scored": len(output_rows),
            "missing_scores": missing_population,
            "development_review_band": sum(
                row["development_review_band"] for row in output_rows
            ),
            "score_quantiles": {
                str(q): float(np.quantile(probabilities, q))
                for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
            },
        },
        "artifact_sha256": {
            "benchmark_manifest": sha256(args.benchmark_manifest),
            "benchmark_scores": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.benchmark_scores
            ],
            "population_manifest": sha256(args.population_manifest),
            "population_scores": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.population_scores
            ],
            "output": sha256(args.output),
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
