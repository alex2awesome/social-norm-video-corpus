#!/usr/bin/env python3
"""Benchmark cheap visual features on V15+V17 and transfer once to V18.

This is read-only, post-hoc development. Thresholds and classifiers are fit
only on V15+V17; V18 remains a source-disjoint transfer cohort. No output is a
keep/reject authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.benchmark_instructional_storyboard_clip import (
        choose_threshold,
        metrics,
        oof_logistic,
        sha256,
    )
except ModuleNotFoundError:
    from benchmark_instructional_storyboard_clip import (
        choose_threshold,
        metrics,
        oof_logistic,
        sha256,
    )


FEATURE_GROUPS = {
    "temporal": (
        "motion_mean",
        "motion_max",
        "histogram_delta_mean",
        "hard_cut_fraction",
    ),
    "faces": (
        "face_count_mean",
        "face_present_fraction",
        "multiple_faces_fraction",
    ),
    "temporal_plus_faces": (
        "motion_mean",
        "motion_max",
        "histogram_delta_mean",
        "hard_cut_fraction",
        "face_count_mean",
        "face_present_fraction",
        "multiple_faces_fraction",
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def choose_recall_floor(
    labels: np.ndarray,
    values: np.ndarray,
    minimum_recall: float,
) -> float:
    """Choose the highest lower bound that preserves required positive recall."""
    choices = []
    for threshold in np.unique(
        np.concatenate(([float("-inf")], values, [float("inf")]))
    ):
        result = metrics(labels, values >= threshold)
        if result["recall"] is not None and result["recall"] >= minimum_recall:
            choices.append((float(threshold), result["specificity"]))
    return max(choices, key=lambda pair: (pair[0], pair[1]))[0]


def best_precision_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    minimum_predictions: int,
) -> tuple[float, dict[str, Any]]:
    """Choose the highest-precision OOF threshold with minimum support."""
    choices = []
    for threshold in np.unique(scores):
        result = metrics(labels, scores >= threshold)
        if result["accepted"] >= minimum_predictions:
            choices.append(
                (
                    float(result["precision"]),
                    float(result["recall"]),
                    float(threshold),
                    result,
                )
            )
    if not choices:
        raise ValueError("no threshold has the requested prediction support")
    _, _, threshold, result = max(choices)
    return threshold, result


def keyed_scores(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Merge score files in order, allowing later codec retries to replace errors."""
    output = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get("error") is None and row.get("item_id"):
                output[str(row["item_id"])] = row
    return output


def score_transfer(
    values: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    groups: np.ndarray,
    minimum_precision: float,
) -> dict[str, Any]:
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
    threshold = choose_threshold(labels[train], oof, minimum_precision)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        ),
    )
    classifier.fit(values[train], labels[train])
    test_scores = classifier.predict_proba(values[~train])[:, 1]
    operating_points = {}
    for minimum_predictions in (3, 5, 10, 20):
        point_threshold, train_metrics = best_precision_threshold(
            labels[train],
            oof,
            minimum_predictions,
        )
        operating_points[str(minimum_predictions)] = {
            "minimum_train_predictions": minimum_predictions,
            "threshold_fit_train_oof": point_threshold,
            "train_oof_metrics": train_metrics,
            "v18_transfer_metrics": metrics(
                labels[~train],
                test_scores >= point_threshold,
            ),
        }
    return {
        "selected_c": c_value,
        "train_oof_average_precision": train_ap,
        "train_oof_roc_auc": roc_auc_score(labels[train], oof),
        "train_selected_threshold": threshold,
        "train_oof_metrics": metrics(
            labels[train], oof >= threshold
        ),
        "v18_average_precision": average_precision_score(
            labels[~train], test_scores
        ),
        "v18_roc_auc": roc_auc_score(labels[~train], test_scores),
        "v18_transfer_metrics": metrics(
            labels[~train], test_scores >= threshold
        ),
        "best_precision_operating_points": operating_points,
        "train_oof_scores": oof.tolist(),
        "v18_scores": test_scores.tolist(),
    }


def benchmark(
    manifest_rows: list[dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
    minimum_precision: float,
    static_minimum_recall: float,
) -> dict[str, Any]:
    if len(manifest_rows) != 168:
        raise ValueError(f"expected 168 benchmark rows, got {len(manifest_rows)}")
    missing = {
        str(row["item_id"]) for row in manifest_rows
    } - set(score_rows)
    if missing:
        raise ValueError(f"missing {len(missing)} successful score rows")

    train = np.asarray(
        [row["audit_cohort"] in {"v15", "v17"} for row in manifest_rows]
    )
    groups = np.asarray([row["uid"] for row in manifest_rows])
    targets = {
        "visual": np.asarray(
            [bool(row["gold_scene_visible"]) for row in manifest_rows]
        ),
        "usable": np.asarray(
            [bool(row["gold_usable"]) for row in manifest_rows]
        ),
        "exact": np.asarray(
            [bool(row["gold_exact_social_norm"]) for row in manifest_rows]
        ),
    }
    matrices = {
        name: np.asarray(
            [
                [
                    float(score_rows[str(row["item_id"])]["low_level"][key])
                    for key in keys
                ]
                for row in manifest_rows
            ],
            dtype=np.float64,
        )
        for name, keys in FEATURE_GROUPS.items()
    }
    model_results = {
        name: {
            target: score_transfer(
                values,
                labels,
                train,
                groups,
                minimum_precision,
            )
            for target, labels in targets.items()
        }
        for name, values in matrices.items()
    }
    static_floors = {}
    for feature in ("motion_mean", "histogram_delta_mean"):
        values = np.asarray(
            [
                float(score_rows[str(row["item_id"])]["low_level"][feature])
                for row in manifest_rows
            ]
        )
        threshold = choose_recall_floor(
            targets["visual"][train],
            values[train],
            static_minimum_recall,
        )
        static_floors[feature] = {
            "threshold_fit_v15_v17": threshold,
            "v15_v17_metrics": metrics(
                targets["visual"][train], values[train] >= threshold
            ),
            "v18_transfer_metrics": metrics(
                targets["visual"][~train], values[~train] >= threshold
            ),
        }
    return {
        "kind": "instructional_v19_cheap_visual_feature_benchmark",
        "status": "posthoc_development_not_promotion",
        "policy": "read_only_shadow_features_no_keep_or_reject_authority",
        "protocol": "fit_v15_v17_test_v18_source_disjoint",
        "coverage": {
            "total": len(manifest_rows),
            "train_v15_v17": int(train.sum()),
            "test_v18": int((~train).sum()),
        },
        "minimum_train_precision": minimum_precision,
        "static_floor_minimum_train_recall": static_minimum_recall,
        "feature_groups": {
            name: list(keys) for name, keys in FEATURE_GROUPS.items()
        },
        "model_results": model_results,
        "static_floor_results": static_floors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--scores",
        type=Path,
        action="append",
        required=True,
        help="JSONL score artifact; repeat for codec retries (later wins)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-train-precision", type=float, default=0.90)
    parser.add_argument("--static-minimum-train-recall", type=float, default=0.95)
    args = parser.parse_args()
    report = benchmark(
        read_jsonl(args.manifest),
        keyed_scores(args.scores),
        args.minimum_train_precision,
        args.static_minimum_train_recall,
    )
    report["artifact_sha256"] = {
        "manifest": sha256(args.manifest),
        "scores": [
            {"path": str(path), "sha256": sha256(path)}
            for path in args.scores
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
