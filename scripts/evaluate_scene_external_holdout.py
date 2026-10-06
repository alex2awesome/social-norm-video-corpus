#!/usr/bin/env python3
"""Apply frozen scene thresholds/rules to a source-disjoint holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_visual_scene_scores import flatten_baseline, metrics  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def apply_threshold(features: dict, rule: dict) -> bool:
    value = features[rule["feature"]]
    if rule["direction"] == "ge":
        return value >= rule["threshold"]
    return value <= rule["threshold"]


def scored_rule(rows: list[dict], name: str, predict) -> dict:
    predictions = [bool(predict(row)) for row in rows]
    report = metrics([row["gold"] for row in rows], predictions)
    report["errors"] = [
        {
            "item_id": row["item_id"],
            "gold": row["gold"],
            "predicted": prediction,
        }
        for row, prediction in zip(rows, predictions)
        if row["gold"] != prediction
    ]
    report["name"] = name
    return report


def social_text_pass(record: dict, strict: bool = False) -> bool:
    result = record["result"]
    if result.get("social_norm_candidate") != "yes":
        return False
    if not strict:
        return True
    return (
        result.get("concrete_behavior_named") == "yes"
        and result.get("affected_other_or_shared_setting_named") == "yes"
        and result.get("norm_type")
        in {"tacit_interpersonal", "shared_public"}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--calibration-metrics", type=Path, required=True)
    parser.add_argument("--qwen-vlm", type=Path)
    parser.add_argument("--glm-vlm", type=Path)
    parser.add_argument("--social-text", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = {r["item_id"]: r for r in load_jsonl(args.manifest)}
    rows = [
        {
            "item_id": record["item_id"],
            "gold": bool(manifest[record["item_id"]]["gold_social_scene_visible"]),
            "label_gold": bool(
                manifest[record["item_id"]]["gold_label_matched_visible"]
            ),
            "features": flatten_baseline(record),
        }
        for record in load_jsonl(args.features)
        if record["item_id"] in manifest
    ]
    calibration = json.loads(args.calibration_metrics.read_text())
    frozen = {
        rule["feature"]: rule
        for rule in calibration["baseline_features"]
    }
    feature_names = {
        "xclip_dialogue_probability": (
            "xclip.probabilities.1:a dialogue scene between characters inside a "
        ),
        "xclip_dialogue_margin": "xclip.derived.dialogue_minus_direct_address",
        "clip_roleplay_mean": (
            "clip.mean_probabilities.0:a staged role play showing a social interacti"
        ),
    }
    report = {"items": len(rows), "frozen_feature_rules": {}, "ensembles": {}}
    for name, feature in feature_names.items():
        rule = frozen[feature]
        report["frozen_feature_rules"][name] = scored_rule(
            rows, name, lambda row, rule=rule: apply_threshold(row["features"], rule)
        )
        report["frozen_feature_rules"][name]["feature"] = feature
        report["frozen_feature_rules"][name]["direction"] = rule["direction"]
        report["frozen_feature_rules"][name]["threshold"] = rule["threshold"]

    margin = frozen[feature_names["xclip_dialogue_margin"]]
    roleplay = frozen[feature_names["clip_roleplay_mean"]]
    report["ensembles"]["cheap_intersection"] = scored_rule(
        rows,
        "cheap_intersection",
        lambda row: apply_threshold(row["features"], margin)
        and apply_threshold(row["features"], roleplay),
    )
    social_text = {}
    if args.social_text:
        social_text = {
            r["item_id"]: r
            for r in load_jsonl(args.social_text)
            if r.get("result")
        }
        text_rows = [row for row in rows if row["item_id"] in social_text]
        report["social_text_items_valid"] = len(text_rows)
        report["ensembles"]["social_text_only"] = scored_rule(
            text_rows,
            "social_text_only",
            lambda row: social_text_pass(social_text[row["item_id"]]),
        )
        report["ensembles"]["social_text_strict"] = scored_rule(
            text_rows,
            "social_text_strict",
            lambda row: social_text_pass(
                social_text[row["item_id"]], strict=True
            ),
        )
        report["ensembles"]["cheap_intersection_and_social_text"] = scored_rule(
            text_rows,
            "cheap_intersection_and_social_text",
            lambda row: apply_threshold(row["features"], margin)
            and apply_threshold(row["features"], roleplay)
            and social_text_pass(social_text[row["item_id"]]),
        )
        report["ensembles"][
            "cheap_intersection_and_social_text_strict"
        ] = scored_rule(
            text_rows,
            "cheap_intersection_and_social_text_strict",
            lambda row: apply_threshold(row["features"], margin)
            and apply_threshold(row["features"], roleplay)
            and social_text_pass(
                social_text[row["item_id"]], strict=True
            ),
        )

    if args.qwen_vlm and args.glm_vlm:
        qwen = {
            r["item_id"]: r
            for r in load_jsonl(args.qwen_vlm)
            if r.get("result")
        }
        glm = {
            r["item_id"]: r
            for r in load_jsonl(args.glm_vlm)
            if r.get("result")
        }
        vlm_rows = [
            row
            for row in rows
            if row["item_id"] in qwen and row["item_id"] in glm
        ]

        def answer(model: dict, row: dict, field: str) -> bool:
            return model[row["item_id"]]["result"].get(field) == "yes"

        report["vlm_items_both_valid"] = len(vlm_rows)
        for field, target in (
            ("situated_social_scenario_visible", "gold"),
            ("proposed_norm_plausibly_demonstrated", "label_gold"),
        ):
            for operator in ("and", "or"):
                key = f"dual_vlm_{field}_{operator}"
                local_rows = [
                    {**row, "gold": row[target]}
                    for row in vlm_rows
                ]
                if operator == "and":
                    fn = lambda row, field=field: answer(qwen, row, field) and answer(
                        glm, row, field
                    )
                else:
                    fn = lambda row, field=field: answer(qwen, row, field) or answer(
                        glm, row, field
                    )
                report["ensembles"][key] = scored_rule(local_rows, key, fn)
                if social_text:
                    text_vlm_rows = [
                        row
                        for row in local_rows
                        if row["item_id"] in social_text
                    ]
                    report["ensembles"][f"{key}_and_social_text"] = scored_rule(
                        text_vlm_rows,
                        f"{key}_and_social_text",
                        lambda row, fn=fn: fn(row)
                        and social_text_pass(social_text[row["item_id"]]),
                    )
                    report["ensembles"][
                        f"{key}_and_social_text_strict"
                    ] = scored_rule(
                        text_vlm_rows,
                        f"{key}_and_social_text_strict",
                        lambda row, fn=fn: fn(row)
                        and social_text_pass(
                            social_text[row["item_id"]], strict=True
                        ),
                    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
