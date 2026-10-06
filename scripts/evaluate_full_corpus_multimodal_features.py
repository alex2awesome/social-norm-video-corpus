#!/usr/bin/env python3
"""Evaluate pose, CLIP, and X-CLIP shadow scores on frozen human labels.

Models are fit on the 60-item mechanism cohort and evaluated on the independent
40-item source-disjoint uniform cohort. High-precision thresholds are selected
only from out-of-fold mechanism predictions. This script is read-only and does
not promote a corpus rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from scripts.evaluate_full_corpus_audit_features import load_gold
else:
    from evaluate_full_corpus_audit_features import load_gold


LOW_LEVEL = (
    "face_count_mean",
    "face_present_fraction",
    "hard_cut_fraction",
    "histogram_delta_mean",
    "motion_max",
    "motion_mean",
    "multiple_faces_fraction",
)
POSE = (
    "close_pair_fraction",
    "largest_person_area_fraction_mean",
    "multiple_people_fraction",
    "nearest_pair_distance_mean",
    "pair_iou_mean",
    "person_box_area_mean",
    "person_center_motion_mean",
    "person_count_mean",
    "person_count_std",
    "person_present_fraction",
    "posed_person_count_mean",
    "second_to_first_area_ratio_mean",
    "stable_single_person_transition_fraction",
    "wrist_near_other_person_fraction",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def last_successful_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Merge append-only attempts, keeping the last successful section values."""
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get("error") is not None or not row.get("item_id"):
                continue
            target = merged.setdefault(str(row["item_id"]), {})
            for section in (
                "low_level",
                "keypoints",
                "clip_scores",
                "xclip_scores",
            ):
                if isinstance(row.get(section), dict) and row[section]:
                    target[section] = row[section]
            if isinstance(row.get("regex"), dict) and row["regex"]:
                target["transcript_regex"] = row["regex"]
            if isinstance(row.get("llm"), dict) and row["llm"]:
                target["transcript_llm"] = row["llm"]
    return merged


def ordered_prompt_features(
    section: dict[str, Any],
    value_fields: tuple[str, ...],
    prefix: str,
) -> dict[str, float]:
    prompts = section.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        return {}
    result: dict[str, float] = {}
    for value_field in value_fields:
        values = section.get(value_field)
        if not isinstance(values, list) or len(values) != len(prompts):
            continue
        for index, value in enumerate(values):
            result[f"{prefix}.{value_field}.{index}"] = float(value)
    return result


def flatten(row: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    low = row.get("low_level") or {}
    pose = row.get("keypoints") or {}
    result.update(
        {f"low.{name}": float(low[name]) for name in LOW_LEVEL if name in low}
    )
    result.update(
        {f"pose.{name}": float(pose[name]) for name in POSE if name in pose}
    )
    result.update(
        ordered_prompt_features(
            row.get("clip_scores") or {},
            ("mean_probabilities", "max_probabilities"),
            "clip",
        )
    )
    result.update(
        ordered_prompt_features(
            row.get("xclip_scores") or {},
            ("probabilities",),
            "xclip",
        )
    )
    regex = row.get("transcript_regex") or {}
    result.update(
        {
            f"transcript.regex.{name}": float(value)
            for name, value in regex.items()
            if isinstance(value, (int, float))
        }
    )
    llm = row.get("transcript_llm") or {}
    for name in ("direct_depiction_prior", "offscreen_description_prior"):
        if isinstance(llm.get(name), (int, float)):
            result[f"transcript.llm.{name}"] = float(llm[name])
    discourse = llm.get("discourse_mode")
    if isinstance(discourse, str):
        for name in (
            "direct_interaction",
            "role_play",
            "instructional_narration",
            "retrospective_account",
            "news_report",
            "lecture",
            "unknown",
        ):
            result[f"transcript.discourse.{name}"] = float(discourse == name)
    return result


def feature_groups(feature_names: set[str]) -> dict[str, list[str]]:
    atomic = {
        "low_level": sorted(name for name in feature_names if name.startswith("low.")),
        "pose": sorted(name for name in feature_names if name.startswith("pose.")),
        "clip": sorted(name for name in feature_names if name.startswith("clip.")),
        "xclip": sorted(name for name in feature_names if name.startswith("xclip.")),
        "transcript_regex": sorted(
            name for name in feature_names
            if name.startswith("transcript.regex.")
        ),
        "transcript_llm": sorted(
            name for name in feature_names
            if name.startswith("transcript.llm.")
            or name.startswith("transcript.discourse.")
        ),
    }
    groups = {name: values for name, values in atomic.items() if values}
    if atomic["low_level"] and atomic["pose"]:
        groups["low_plus_pose"] = sorted(
            atomic["low_level"] + atomic["pose"]
        )
    if atomic["clip"] and atomic["xclip"]:
        groups["clip_plus_xclip"] = sorted(
            atomic["clip"] + atomic["xclip"]
        )
    if atomic["transcript_regex"] and atomic["transcript_llm"]:
        groups["transcript_all"] = sorted(
            atomic["transcript_regex"] + atomic["transcript_llm"]
        )
    groups["all"] = sorted(
        atomic["low_level"]
        + atomic["pose"]
        + atomic["clip"]
        + atomic["xclip"]
        + atomic["transcript_regex"]
        + atomic["transcript_llm"]
    )
    return groups


def estimators() -> dict[str, Any]:
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


def scalar_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    prediction = probability >= threshold
    result: dict[str, Any] = {
        "items": int(len(truth)),
        "positives": int(truth.sum()),
        "predicted_positive": int(prediction.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "confusion_matrix": confusion_matrix(
            truth, prediction, labels=[0, 1]
        ).tolist(),
    }
    if len(set(truth.tolist())) == 2:
        result["roc_auc"] = float(roc_auc_score(truth, probability))
        result["average_precision"] = float(
            average_precision_score(truth, probability)
        )
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def select_oof_threshold(
    truth: np.ndarray,
    probability: np.ndarray,
    minimum_precision: float = 0.8,
    minimum_predictions: int = 3,
) -> dict[str, Any]:
    candidates = sorted({float(value) for value in probability}, reverse=True)
    eligible = []
    for threshold in candidates:
        metrics = scalar_metrics(truth, probability, threshold)
        if (
            metrics["predicted_positive"] >= minimum_predictions
            and metrics["precision"] >= minimum_precision
        ):
            eligible.append(metrics)
    if not eligible:
        return {
            "available": False,
            "minimum_precision": minimum_precision,
            "minimum_predictions": minimum_predictions,
        }
    # Maximize recall, then precision, and prefer the higher threshold on ties.
    selected = max(
        eligible,
        key=lambda row: (row["recall"], row["precision"], row["threshold"]),
    )
    return {
        "available": True,
        "minimum_precision": minimum_precision,
        "minimum_predictions": minimum_predictions,
        "selection_metrics": selected,
    }


def evaluate_model(
    estimator: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    minimum_class = int(np.bincount(y_train, minlength=2).min())
    folds = min(5, minimum_class)
    result: dict[str, Any] = {}
    threshold = None
    if folds >= 2:
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=0,
        )
        oof = cross_val_predict(
            clone(estimator),
            x_train,
            y_train,
            cv=splitter,
            method="predict_proba",
        )[:, 1]
        selected = select_oof_threshold(y_train, oof)
        result["oof_threshold_selection"] = selected
        if selected["available"]:
            threshold = float(selected["selection_metrics"]["threshold"])
    else:
        result["oof_threshold_selection"] = {
            "available": False,
            "reason": "fewer_than_two_examples_in_smallest_training_class",
        }

    fitted = clone(estimator).fit(x_train, y_train)
    probability = fitted.predict_proba(x_test)[:, 1]
    result["uniform_at_0_5"] = scalar_metrics(
        y_test, probability, threshold=0.5
    )
    if threshold is not None:
        result["uniform_at_oof_high_precision_threshold"] = scalar_metrics(
            y_test, probability, threshold=threshold
        )
    return result


def matrix(
    ids: list[str],
    flat: dict[str, dict[str, float]],
    names: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [flat[item_id].get(name, np.nan) for name in names]
            for item_id in ids
        ],
        dtype=float,
    )


def evaluate_pillar(
    audit_root: Path,
    pillar: str,
    flat: dict[str, dict[str, float]],
) -> dict[str, Any]:
    train = load_gold(audit_root, "mechanism", pillar)
    test = load_gold(audit_root, "uniform", pillar)
    missing = sorted((set(train) | set(test)) - set(flat))
    if missing:
        raise ValueError(f"{pillar}: missing multimodal rows: {missing}")
    train_ids = sorted(train)
    test_ids = sorted(test)
    available_names = set.union(
        *(set(flat[item_id]) for item_id in train_ids + test_ids)
    )
    result: dict[str, Any] = {}
    for group_name, names in feature_groups(available_names).items():
        group: dict[str, Any] = {
            "features": names,
            "targets": {},
        }
        x_train = matrix(train_ids, flat, names)
        x_test = matrix(test_ids, flat, names)
        for target in ("visual", "strict"):
            y_train = np.asarray(
                [train[item_id][target] for item_id in train_ids],
                dtype=int,
            )
            y_test = np.asarray(
                [test[item_id][target] for item_id in test_ids],
                dtype=int,
            )
            target_result: dict[str, Any] = {
                "training_positives": int(y_train.sum()),
                "uniform_positives": int(y_test.sum()),
                "models": {},
            }
            if len(set(y_train.tolist())) < 2:
                target_result["error"] = "training cohort contains one class"
            else:
                for model_name, estimator in estimators().items():
                    target_result["models"][model_name] = evaluate_model(
                        estimator,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                    )
            group["targets"][target] = target_result
        result[group_name] = group
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path, action="append")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    merged = last_successful_rows(args.features)
    flat = {item_id: flatten(row) for item_id, row in merged.items()}
    report = {
        "protocol": "fit_mechanism_60_test_uniform_source_disjoint_40",
        "threshold_protocol": (
            "high_precision_threshold_selected_from_mechanism_oof_only"
        ),
        "policy": "read_only_no_rule_promotion",
        "feature_files": [str(path) for path in args.features],
        "items": len(flat),
        "pillars": {
            pillar: evaluate_pillar(args.audit_root, pillar, flat)
            for pillar in ("instructional", "witnessed", "commentary")
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
