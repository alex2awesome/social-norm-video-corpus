#!/usr/bin/env python3
"""Evaluate shadow scene scores against frozen manual judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split(uid: str) -> str:
    value = int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if value < 0.60 else "test"


def metrics(labels: list[bool], predictions: list[bool]) -> dict:
    tp = sum(y and p for y, p in zip(labels, predictions))
    tn = sum((not y) and (not p) for y, p in zip(labels, predictions))
    fp = sum((not y) and p for y, p in zip(labels, predictions))
    fn = sum(y and (not p) for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": len(labels),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    }


def flatten_baseline(record: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for section in ("low_level", "keypoints"):
        for key, value in (record.get(section) or {}).items():
            if isinstance(value, (int, float)):
                values[f"{section}.{key}"] = float(value)
    clip = record.get("clip_scores") or {}
    prompts = clip.get("prompts") or []
    for aggregate in ("mean_probabilities", "max_probabilities"):
        for index, value in enumerate(clip.get(aggregate) or []):
            short = prompts[index].split(",")[0][:45] if index < len(prompts) else str(index)
            values[f"clip.{aggregate}.{index}:{short}"] = float(value)
    xclip = record.get("xclip_scores") or {}
    xclip_prompts = xclip.get("prompts") or []
    for aggregate in ("logits", "probabilities"):
        for index, value in enumerate(xclip.get(aggregate) or []):
            short = (
                xclip_prompts[index].split(",")[0][:45]
                if index < len(xclip_prompts)
                else str(index)
            )
            values[f"xclip.{aggregate}.{index}:{short}"] = float(value)
    xclip_probs = xclip.get("probabilities") or []
    if len(xclip_probs) >= 10:
        positive = [float(xclip_probs[index]) for index in range(5)]
        negative = [float(xclip_probs[index]) for index in range(5, 10)]
        values["xclip.derived.social_scene_probability_sum"] = sum(positive)
        values["xclip.derived.social_minus_nonsocial_sum"] = (
            sum(positive) - sum(negative)
        )
        values["xclip.derived.dialogue_minus_direct_address"] = float(
            xclip_probs[1]
        ) - max(float(xclip_probs[5]), float(xclip_probs[6]))
        values["xclip.derived.best_social_minus_best_nonsocial"] = (
            max(positive) - max(negative)
        )
    clip_mean = clip.get("mean_probabilities") or []
    clip_max = clip.get("max_probabilities") or []
    if len(clip_mean) >= 5:
        values["clip.derived.roleplay_minus_slide_mean"] = float(
            clip_mean[0]
        ) - float(clip_mean[4])
    if len(clip_max) >= 5:
        values["clip.derived.roleplay_minus_slide_max"] = float(
            clip_max[0]
        ) - float(clip_max[4])
    return values


def flatten_transcript(record: dict, prefix: str = "") -> dict[str, float]:
    values = {
        f"{prefix}regex.{key}": float(value)
        for key, value in (record.get("regex") or {}).items()
        if isinstance(value, (int, float))
    }
    llm = record.get("llm") or {}
    for key in ("direct_depiction_prior", "offscreen_description_prior"):
        if isinstance(llm.get(key), (int, float)):
            values[f"{prefix}llm.{key}"] = float(llm[key])
    return values


def best_threshold(rows: list[dict], feature: str) -> dict | None:
    train = [r for r in rows if split(r["uid"]) == "train" and feature in r["features"]]
    test = [r for r in rows if split(r["uid"]) == "test" and feature in r["features"]]
    if len(train) < 10 or len(test) < 5:
        return None
    candidates = sorted({r["features"][feature] for r in train})
    if len(candidates) > 100:
        candidates = list(np.quantile(candidates, np.linspace(0, 1, 101)))
    best = None
    for direction in ("ge", "le"):
        for threshold in candidates:
            preds = [
                (r["features"][feature] >= threshold)
                if direction == "ge"
                else (r["features"][feature] <= threshold)
                for r in train
            ]
            score = metrics([r["gold"] for r in train], preds)["balanced_accuracy"]
            candidate = (score, direction, threshold)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    _, direction, threshold = best
    predict = lambda r: (
        r["features"][feature] >= threshold
        if direction == "ge"
        else r["features"][feature] <= threshold
    )
    return {
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "train": metrics(
            [r["gold"] for r in train], [predict(r) for r in train]
        ),
        "test": metrics([r["gold"] for r in test], [predict(r) for r in test]),
        "test_errors": [
            {
                "item_id": r["item_id"],
                "uid": r["uid"],
                "pillar": r["pillar"],
                "gold": r["gold"],
                "predicted": predict(r),
                "value": r["features"][feature],
            }
            for r in test
            if predict(r) != r["gold"]
        ],
    }


def evaluate_vlm(path: Path) -> dict:
    rows = [r for r in load_jsonl(path) if r.get("result")]
    rubric = rows[0].get("rubric", "v1") if rows else "v1"
    field_pillars: dict[str, str | None] = {}
    if rubric == "v4":
        fields = {
            "situated_social_scenario_visible": "gold_social_scene_visible",
            "observable_social_behavior_or_speech": "gold_social_scene_visible",
            "usable_demo_after_relabel": "gold_social_scene_visible",
            "proposed_norm_supported": "gold_label_matched_visible",
            "instructional_demo_present": "gold_social_scene_visible",
            "witnessed_action_then_reaction_organic": "gold_label_matched_visible",
            "commentary_event_footage_present": "gold_social_scene_visible",
        }
        field_pillars.update(
            {
                "instructional_demo_present": "instructional",
                "witnessed_action_then_reaction_organic": "witnessed",
                "commentary_event_footage_present": "commentary",
            }
        )
        conjunction_fields = (
            "usable_demo_after_relabel",
            "proposed_norm_supported",
        )
    elif rubric == "v3":
        fields = {
            "situated_social_scenario_visible": "gold_social_scene_visible",
            "characters_interact_or_act_in_shared_context": "gold_social_scene_visible",
            "concrete_action_or_situated_speech_visible": "gold_social_scene_visible",
            "proposed_norm_plausibly_demonstrated": "gold_label_matched_visible",
            "witnessed_violation_action_visible": "gold_label_matched_visible",
        }
        field_pillars["witnessed_violation_action_visible"] = "witnessed"
        conjunction_fields = (
            "situated_social_scenario_visible",
            "proposed_norm_plausibly_demonstrated",
        )
    elif rubric == "v2":
        fields = {
            "concrete_behavior_event_visible": "gold_social_scene_visible",
            "proposed_label_event_visible": "gold_label_matched_visible",
        }
        conjunction_fields = (
            "concrete_behavior_event_visible",
            "proposed_label_event_visible",
        )
    else:
        fields = {
            "physical_social_interaction_visible": "gold_social_scene_visible",
            "action_directly_observable": "gold_scene_visible",
            "labeled_event_directly_depicted": "gold_label_matched_visible",
        }
        conjunction_fields = (
            "physical_social_interaction_visible",
            "action_directly_observable",
            "labeled_event_directly_depicted",
        )
    report = {}
    for field, target in fields.items():
        required_pillar = field_pillars.get(field)
        usable = [
            r
            for r in rows
            if r["result"].get(field) in {"yes", "no"}
            and (required_pillar is None or r["pillar"] == required_pillar)
        ]
        field_report = metrics(
            [bool(r[target]) for r in usable],
            [r["result"][field] == "yes" for r in usable],
        )
        field_report["target"] = target
        field_report["by_pillar"] = {
            pillar: metrics(
                [bool(r[target]) for r in usable if r["pillar"] == pillar],
                [
                    r["result"][field] == "yes"
                    for r in usable
                    if r["pillar"] == pillar
                ],
            )
            for pillar in sorted({r["pillar"] for r in usable})
        }
        field_report["errors"] = [
            {
                "item_id": r["item_id"],
                "uid": r["uid"],
                "pillar": r["pillar"],
                "gold": bool(r[target]),
                "predicted": r["result"][field] == "yes",
                "evidence": r["result"].get("evidence"),
            }
            for r in usable
            if bool(r[target]) != (r["result"][field] == "yes")
        ]
        report[field] = field_report
    combined = [
        r
        for r in rows
        if all(
            r["result"].get(field) in {"yes", "no"}
            for field in conjunction_fields
        )
    ]
    report["strict_conjunction"] = metrics(
        [bool(r["gold_label_matched_visible"]) for r in combined],
        [
            all(
                r["result"][field] == "yes"
                for field in conjunction_fields
            )
            for r in combined
        ],
    )
    report["errors"] = sum(r.get("result") is None for r in load_jsonl(path))
    report["items"] = len(rows)
    report["rubric"] = rubric
    return report


def evaluate_transcript(path: Path) -> dict:
    records = load_jsonl(path)
    rows = []
    for record in records:
        features = flatten_transcript(record)
        rows.append(
            {
                "item_id": record["item_id"],
                "uid": record["uid"],
                "pillar": record["pillar"],
                "gold": bool(record["gold_social_scene_visible"]),
                "features": features,
            }
        )
    features = sorted({key for row in rows for key in row["features"]})
    scored = [best_threshold(rows, feature) for feature in features]
    return {
        "items": len(rows),
        "errors": sum(bool(record.get("error")) for record in records),
        "features": sorted(
            [value for value in scored if value],
            key=lambda value: value["test"]["balanced_accuracy"],
            reverse=True,
        ),
    }


def evaluate_multivariate(rows: list[dict]) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    prefixes = (
        "low_level.motion_mean",
        "low_level.histogram_delta_mean",
        "low_level.hard_cut_fraction",
        "keypoints.person_present_fraction",
        "keypoints.multiple_people_fraction",
        "keypoints.person_count_mean",
        "clip.mean_probabilities.",
        "clip.max_probabilities.",
        "clip.derived.",
        "xclip.logits.",
        "xclip.probabilities.",
        "xclip.derived.",
        "transcript.",
    )
    features = sorted(
        {
            key
            for row in rows
            for key in row["features"]
            if key.startswith(prefixes)
        }
    )
    train = [row for row in rows if split(row["uid"]) == "train"]
    test = [row for row in rows if split(row["uid"]) == "test"]
    if not features:
        return {"features": [], "models": {}, "error": "no eligible features"}

    def matrix(selected: list[dict]) -> np.ndarray:
        return np.asarray(
            [
                [row["features"].get(feature, np.nan) for feature in features]
                for row in selected
            ],
            dtype=float,
        )

    x_train, x_test = matrix(train), matrix(test)
    y_train = np.asarray([row["gold"] for row in train], dtype=int)
    y_test = np.asarray([row["gold"] for row in test], dtype=int)
    models = {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", C=0.25, max_iter=2000, random_state=0
            ),
        ),
        "random_forest_shallow": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=0,
            ),
        ),
    }
    report = {"features": features, "models": {}}
    for name, model in models.items():
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train).astype(bool).tolist()
        test_pred = model.predict(x_test).astype(bool).tolist()
        report["models"][name] = {
            "train": metrics(y_train.astype(bool).tolist(), train_pred),
            "test": metrics(y_test.astype(bool).tolist(), test_pred),
            "test_errors": [
                {
                    "item_id": row["item_id"],
                    "uid": row["uid"],
                    "pillar": row["pillar"],
                    "gold": bool(gold),
                    "predicted": bool(predicted),
                }
                for row, gold, predicted in zip(test, y_test, test_pred)
                if bool(gold) != bool(predicted)
            ],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--vlm", type=Path, action="append", default=[])
    parser.add_argument("--transcript", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report: dict = {"baseline_features": [], "vlm": {}}
    if args.baselines and args.baselines.exists():
        records = load_jsonl(args.baselines)
        manifest_by_id = (
            {row["item_id"]: row for row in load_jsonl(args.manifest)}
            if args.manifest and args.manifest.exists()
            else {}
        )
        rows = [
            {
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": row["pillar"],
                "gold": bool(
                    manifest_by_id.get(row["item_id"], row).get(
                        "gold_social_scene_visible", row["gold_scene_visible"]
                    )
                ),
                "features": flatten_baseline(row),
            }
            for row in records
        ]
        if args.transcript:
            transcript_by_id = {
                record["item_id"]: record
                for record in load_jsonl(args.transcript[0])
            }
            for row in rows:
                if row["item_id"] in transcript_by_id:
                    row["features"].update(
                        flatten_transcript(
                            transcript_by_id[row["item_id"]], prefix="transcript."
                        )
                    )
        features = sorted({key for row in rows for key in row["features"]})
        scored = [best_threshold(rows, feature) for feature in features]
        report["baseline_features"] = sorted(
            [value for value in scored if value],
            key=lambda value: value["test"]["balanced_accuracy"],
            reverse=True,
        )
        report["baseline_items"] = len(rows)
        report["split_counts"] = dict(
            (name, sum(split(row["uid"]) == name for row in rows))
            for name in ("train", "test")
        )
        report["multivariate"] = evaluate_multivariate(rows)

    for path in args.vlm:
        if path.exists():
            report["vlm"][path.name] = evaluate_vlm(path)
    report["transcript"] = {}
    for path in args.transcript:
        if path.exists():
            report["transcript"][path.name] = evaluate_transcript(path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
