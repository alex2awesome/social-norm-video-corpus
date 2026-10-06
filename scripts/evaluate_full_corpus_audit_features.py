#!/usr/bin/env python3
"""Evaluate cheap shadow features against frozen source-disjoint human audits.

This script is read-only. It fits on the 60-item mechanism cohort and reports
performance on the independently sampled 40-item uniform cohort. It never
changes corpus metadata or promotes a rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "face_count_mean",
    "face_present_fraction",
    "hard_cut_fraction",
    "histogram_delta_mean",
    "motion_max",
    "motion_mean",
    "multiple_faces_fraction",
)
STRICT_FIELD = {
    "instructional": "strict_instructional_pass",
    "witnessed": "strict_witnessed_pass",
    "commentary": "strict_commentary_visual_pass",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row["item_id"])
        if item_id in result:
            raise ValueError(f"duplicate item_id: {item_id}")
        result[item_id] = row
    return result


def final_visual_label(
    item_id: str,
    sparse: dict[str, Any],
    dense: dict[str, dict[str, Any]],
) -> int:
    if sparse["dense_review_required"] == "yes":
        row = dense[item_id]
        scene = row["dense_scene_visible"]
        action = row["dense_concrete_action_visible"]
    else:
        scene = sparse["situated_social_scene"]
        action = sparse["concrete_behavior_visible"]
    return int(scene == "yes" and action == "yes")


def load_gold(
    audit_root: Path,
    cohort: str,
    pillar: str,
) -> dict[str, dict[str, int]]:
    root = audit_root / cohort / pillar
    sparse_rows = read_jsonl(root / "manual_blind_review.jsonl")
    dense_path = root / "dense_followup" / "manual_dense_review.jsonl"
    dense = keyed(read_jsonl(dense_path)) if dense_path.exists() else {}
    semantic = keyed(read_jsonl(root / "manual_post_reveal_review.jsonl"))
    result: dict[str, dict[str, int]] = {}
    for sparse in sparse_rows:
        item_id = str(sparse["item_id"])
        result[item_id] = {
            "visual": final_visual_label(item_id, sparse, dense),
            "strict": int(semantic[item_id][STRICT_FIELD[pillar]] == "yes"),
        }
    if len(result) != len(sparse_rows):
        raise ValueError(f"{cohort}/{pillar}: duplicate gold rows")
    return result


def load_features(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for row in read_jsonl(path):
        values = row.get("low_level")
        if row.get("error") is not None or not isinstance(values, dict):
            continue
        result[str(row["item_id"])] = np.asarray(
            [float(values.get(name, np.nan)) for name in FEATURES],
            dtype=float,
        )
    return result


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    negatives = int((y_true == 0).sum())
    positives = int((y_true == 1).sum())
    result: dict[str, Any] = {
        "items": int(len(y_true)),
        "positives": positives,
        "negatives": negatives,
        "precision_at_0_5": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "recall_at_0_5": float(recall_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, prediction, labels=[0, 1]
        ).tolist(),
    }
    if positives and negatives:
        result["roc_auc"] = float(roc_auc_score(y_true, probability))
        result["average_precision"] = float(
            average_precision_score(y_true, probability)
        )
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def models() -> dict[str, Any]:
    return {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=0,
            ),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=0,
                n_jobs=1,
            ),
        ),
    }


def evaluate_pillar(
    audit_root: Path,
    pillar: str,
    feature_path: Path,
) -> dict[str, Any]:
    train = load_gold(audit_root, "mechanism", pillar)
    test = load_gold(audit_root, "uniform", pillar)
    features = load_features(feature_path)
    missing_train = sorted(set(train) - set(features))
    missing_test = sorted(set(test) - set(features))
    if missing_train or missing_test:
        raise ValueError(
            f"{pillar}: missing features; train={missing_train}, test={missing_test}"
        )
    train_ids = sorted(train)
    test_ids = sorted(test)
    x_train = np.stack([features[item_id] for item_id in train_ids])
    x_test = np.stack([features[item_id] for item_id in test_ids])
    result: dict[str, Any] = {
        "train_cohort": "mechanism",
        "train_items": len(train_ids),
        "test_cohort": "uniform_source_disjoint",
        "test_items": len(test_ids),
        "features": list(FEATURES),
        "targets": {},
    }
    for target in ("visual", "strict"):
        y_train = np.asarray([train[item_id][target] for item_id in train_ids])
        y_test = np.asarray([test[item_id][target] for item_id in test_ids])
        target_result: dict[str, Any] = {
            "train_positives": int(y_train.sum()),
            "test_positives": int(y_test.sum()),
            "models": {},
        }
        if len(set(y_train.tolist())) < 2:
            target_result["error"] = "training cohort contains one class"
        else:
            for name, model in models().items():
                model.fit(x_train, y_train)
                probability = model.predict_proba(x_test)[:, 1]
                target_result["models"][name] = metrics(y_test, probability)
        result["targets"][target] = target_result
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--instructional-features", required=True, type=Path)
    parser.add_argument("--witnessed-features", required=True, type=Path)
    parser.add_argument("--commentary-features", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "protocol": "fit_mechanism_60_test_uniform_source_disjoint_40",
        "policy": "read_only_no_rule_promotion",
        "pillars": {
            "instructional": evaluate_pillar(
                args.audit_root, "instructional", args.instructional_features
            ),
            "witnessed": evaluate_pillar(
                args.audit_root, "witnessed", args.witnessed_features
            ),
            "commentary": evaluate_pillar(
                args.audit_root, "commentary", args.commentary_features
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
