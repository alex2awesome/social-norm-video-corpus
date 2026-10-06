#!/usr/bin/env python3
"""Test predeclared VLM/pose/CLIP conjunctions on the frozen uniform holdout.

The visual classifiers are fit only on the 60-item mechanism cohort. The
source-disjoint 40-item uniform cohort is used once for reporting. Outputs are
read-only and include every selected/error item for manual reconciliation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

if __package__:
    from scripts.evaluate_full_corpus_audit_features import load_gold
    from scripts.evaluate_full_corpus_multimodal_features import (
        estimators,
        flatten,
        last_successful_rows,
        matrix,
        select_oof_threshold,
    )
    from scripts.evaluate_full_corpus_vlm_benchmark import (
        latest_successes,
        v5_clean_demo,
        v5_exact_label,
        v7c_strict_witnessed,
    )
else:
    from evaluate_full_corpus_audit_features import load_gold
    from evaluate_full_corpus_multimodal_features import (
        estimators,
        flatten,
        last_successful_rows,
        matrix,
        select_oof_threshold,
    )
    from evaluate_full_corpus_vlm_benchmark import (
        latest_successes,
        v5_clean_demo,
        v5_exact_label,
        v7c_strict_witnessed,
    )


def rule_metrics(
    item_ids: list[str],
    truth: dict[str, int],
    selected: dict[str, bool],
) -> dict[str, Any]:
    gold = np.asarray([truth[item_id] for item_id in item_ids], dtype=int)
    prediction = np.asarray(
        [int(bool(selected[item_id])) for item_id in item_ids],
        dtype=int,
    )
    tn, fp, fn, tp = confusion_matrix(
        gold, prediction, labels=[0, 1]
    ).ravel()
    return {
        "items": len(item_ids),
        "gold_positive": int(gold.sum()),
        "selected": int(prediction.sum()),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "balanced_accuracy": float(
            balanced_accuracy_score(gold, prediction)
        ),
        "selected_item_ids": [
            item_id for item_id in item_ids if selected[item_id]
        ],
        "false_positive_item_ids": [
            item_id
            for item_id in item_ids
            if selected[item_id] and not truth[item_id]
        ],
        "false_negative_item_ids": [
            item_id
            for item_id in item_ids
            if not selected[item_id] and truth[item_id]
        ],
    }


def fitted_probabilities(
    train_ids: list[str],
    test_ids: list[str],
    flat: dict[str, dict[str, float]],
    truth: dict[str, int],
    prefixes: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, Any]]:
    names = sorted(
        {
            name
            for item_id in train_ids + test_ids
            for name in flat[item_id]
            if name.startswith(prefixes)
        }
    )
    x_train = matrix(train_ids, flat, names)
    x_test = matrix(test_ids, flat, names)
    y_train = np.asarray([truth[item_id] for item_id in train_ids], dtype=int)
    estimator = estimators()["logistic"]
    fitted = clone(estimator).fit(x_train, y_train)
    probabilities = fitted.predict_proba(x_test)[:, 1]

    smallest_class = int(np.bincount(y_train, minlength=2).min())
    threshold_result: dict[str, Any] = {
        "available": False,
        "reason": "fewer_than_two_examples_in_smallest_training_class",
    }
    if smallest_class >= 2:
        folds = min(5, smallest_class)
        oof = cross_val_predict(
            clone(estimator),
            x_train,
            y_train,
            cv=StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=0,
            ),
            method="predict_proba",
        )[:, 1]
        threshold_result = select_oof_threshold(y_train, oof)
    return (
        dict(zip(test_ids, map(float, probabilities))),
        {"features": names, "oof_high_precision_threshold": threshold_result},
    )


def evaluate_pillar(
    audit_root: Path,
    pillar: str,
    flat: dict[str, dict[str, float]],
    v5: dict[str, dict[str, Any]],
    v7c: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    train = load_gold(audit_root, "mechanism", pillar)
    uniform = load_gold(audit_root, "uniform", pillar)
    train_ids = sorted(train)
    test_ids = sorted(uniform)
    missing = sorted((set(train_ids) | set(test_ids)) - set(flat))
    if missing:
        raise ValueError(f"{pillar}: missing features: {missing}")
    if any(item_id not in v5 for item_id in test_ids):
        raise ValueError(f"{pillar}: missing v5 predictions")

    visual_train = {
        item_id: int(values["visual"])
        for item_id, values in train.items()
    }
    visual_truth = {
        item_id: int(values["visual"])
        for item_id, values in uniform.items()
    }
    strict_truth = {
        item_id: int(values["strict"])
        for item_id, values in uniform.items()
    }
    static, static_meta = fitted_probabilities(
        train_ids,
        test_ids,
        flat,
        visual_train,
        ("low.", "pose.", "clip."),
    )
    low_pose, low_pose_meta = fitted_probabilities(
        train_ids,
        test_ids,
        flat,
        visual_train,
        ("low.", "pose."),
    )
    qwen_clean = {
        item_id: v5_clean_demo(v5[item_id]["result"])
        for item_id in test_ids
    }
    qwen_exact = {
        item_id: v5_exact_label(v5[item_id]["result"])
        for item_id in test_ids
    }
    static_half = {
        item_id: static[item_id] >= 0.5 for item_id in test_ids
    }
    low_pose_half = {
        item_id: low_pose[item_id] >= 0.5 for item_id in test_ids
    }
    rules: dict[str, tuple[str, dict[str, bool]]] = {
        "qwen_v5_clean_demo": ("visual", qwen_clean),
        "static_visual_at_0_5": ("visual", static_half),
        "qwen_clean_and_static_visual": (
            "visual",
            {
                item_id: qwen_clean[item_id] and static_half[item_id]
                for item_id in test_ids
            },
        ),
        "qwen_exact_and_static_visual": (
            "strict",
            {
                item_id: qwen_exact[item_id] and static_half[item_id]
                for item_id in test_ids
            },
        ),
    }
    if pillar == "witnessed":
        threshold_record = low_pose_meta["oof_high_precision_threshold"]
        threshold = (
            float(threshold_record["selection_metrics"]["threshold"])
            if threshold_record.get("available")
            else 0.5
        )
        low_pose_oof = {
            item_id: low_pose[item_id] >= threshold for item_id in test_ids
        }
        if any(item_id not in v7c for item_id in test_ids):
            raise ValueError("witnessed: missing v7c predictions")
        qwen_witnessed = {
            item_id: v7c_strict_witnessed(v7c[item_id]["result"])
            for item_id in test_ids
        }
        rules.update(
            {
                "low_pose_oof_high_precision_scene": (
                    "visual",
                    low_pose_oof,
                ),
                "qwen_clean_and_low_pose_scene": (
                    "visual",
                    {
                        item_id: qwen_clean[item_id]
                        and low_pose_oof[item_id]
                        for item_id in test_ids
                    },
                ),
                "qwen_v7c_and_low_pose_scene": (
                    "strict",
                    {
                        item_id: qwen_witnessed[item_id]
                        and low_pose_oof[item_id]
                        for item_id in test_ids
                    },
                ),
            }
        )
    result = {
        "static_visual_model": static_meta,
        "low_pose_visual_model": low_pose_meta,
        "rules": {},
    }
    for name, (target, selected) in rules.items():
        truth = visual_truth if target == "visual" else strict_truth
        result["rules"][name] = {
            "target": target,
            **rule_metrics(test_ids, truth, selected),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path, action="append")
    parser.add_argument("--v5", required=True, type=Path)
    parser.add_argument("--v7c", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    merged = last_successful_rows(args.features)
    flat = {item_id: flatten(row) for item_id, row in merged.items()}
    v5 = latest_successes(args.v5)
    v7c = latest_successes(args.v7c)
    report = {
        "protocol": "fit_mechanism_60_test_uniform_source_disjoint_40",
        "policy": "shadow_only_every_selected_item_already_manually_audited",
        "candidate_rules_predeclared_in_script": True,
        "pillars": {
            pillar: evaluate_pillar(
                args.audit_root,
                pillar,
                flat,
                v5,
                v7c,
            )
            for pillar in ("instructional", "witnessed", "commentary")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
