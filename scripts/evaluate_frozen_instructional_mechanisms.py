#!/usr/bin/env python3
"""Evaluate prespecified scene mechanisms on frozen instructional gold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_visual_scene_scores import flatten_baseline, metrics  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def valid_by_id(path: Path) -> dict[str, dict]:
    return {
        row["item_id"]: row
        for row in load_jsonl(path)
        if row.get("result") is not None and not row.get("error")
    }


def strict_v5(result: dict, *, exact: bool = False) -> bool:
    passed = all(
        result.get(key) == "yes"
        for key in (
            "social_norm_domain",
            "behavior_occurs_in_scene",
            "affected_party_or_shared_setting_visible",
            "situated_interaction_complete",
            "usable_demo_after_relabel",
        )
    )
    return passed and (
        not exact or result.get("proposed_norm_supported") == "yes"
    )


def text_pass(result: dict, *, strict: bool = False) -> bool:
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


def scored(
    manifest: dict[str, dict],
    ids: list[str],
    target: str,
    predict,
) -> dict:
    predictions = [bool(predict(item_id)) for item_id in ids]
    report = metrics(
        [bool(manifest[item_id][target]) for item_id in ids],
        predictions,
    )
    report["target"] = target
    report["selected"] = [
        {
            "item_id": item_id,
            "audit_index": manifest[item_id]["audit_index"],
            "uid": manifest[item_id]["uid"],
            "gold_disposition": manifest[item_id]["gold_disposition"],
            "correct": bool(manifest[item_id][target]) == prediction,
        }
        for item_id, prediction in zip(ids, predictions)
        if prediction
    ]
    report["errors"] = [
        {
            "item_id": item_id,
            "audit_index": manifest[item_id]["audit_index"],
            "uid": manifest[item_id]["uid"],
            "gold_disposition": manifest[item_id]["gold_disposition"],
            "gold": bool(manifest[item_id][target]),
            "predicted": prediction,
        }
        for item_id, prediction in zip(ids, predictions)
        if bool(manifest[item_id][target]) != prediction
    ]
    return report


def apply_frozen_threshold(value: float, rule: dict) -> bool:
    if rule["direction"] == "ge":
        return value >= rule["threshold"]
    if rule["direction"] == "le":
        return value <= rule["threshold"]
    raise ValueError(f"invalid direction: {rule['direction']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--calibration-metrics", type=Path, required=True)
    parser.add_argument("--qwen-v5", type=Path, required=True)
    parser.add_argument("--glm-v5", type=Path, required=True)
    parser.add_argument("--text-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = {
        row["item_id"]: row for row in load_jsonl(args.manifest)
    }
    for row in manifest.values():
        row["is_social_norm_bool"] = row.get("is_social_norm") == "yes"
    features = {
        row["item_id"]: flatten_baseline(row)
        for row in load_jsonl(args.features)
        if row["item_id"] in manifest
    }
    qwen = valid_by_id(args.qwen_v5)
    glm = valid_by_id(args.glm_v5)
    text = valid_by_id(args.text_gate)
    calibration = json.loads(args.calibration_metrics.read_text())
    frozen = {
        row["feature"]: row for row in calibration["baseline_features"]
    }

    report: dict[str, object] = {
        "items": len(manifest),
        "valid": {
            "features": len(features),
            "qwen_v5": len(qwen),
            "glm_v5": len(glm),
            "text_gate": len(text),
        },
        "rules": {},
    }
    rules: dict[str, dict] = report["rules"]  # type: ignore[assignment]

    for model_name, model in (("qwen", qwen), ("glm", glm)):
        ids = sorted(model)
        rules[f"{model_name}_strict_usable"] = scored(
            manifest,
            ids,
            "gold_usable",
            lambda item_id, model=model: strict_v5(
                model[item_id]["result"]
            ),
        )
        rules[f"{model_name}_strict_exact"] = scored(
            manifest,
            ids,
            "gold_label_matched_visible",
            lambda item_id, model=model: strict_v5(
                model[item_id]["result"], exact=True
            ),
        )

    dual_ids = sorted(set(qwen) & set(glm))
    for operator in ("and", "or"):
        for exact, target, suffix in (
            (False, "gold_usable", "usable"),
            (True, "gold_label_matched_visible", "exact"),
        ):
            def dual_predict(
                item_id: str,
                *,
                exact: bool = exact,
                operator: str = operator,
            ) -> bool:
                answers = (
                    strict_v5(qwen[item_id]["result"], exact=exact),
                    strict_v5(glm[item_id]["result"], exact=exact),
                )
                return all(answers) if operator == "and" else any(answers)

            rules[f"dual_{operator}_strict_{suffix}"] = scored(
                manifest, dual_ids, target, dual_predict
            )

    text_ids = sorted(text)
    for strict in (False, True):
        name = "text_strict" if strict else "text_basic"
        rules[name] = scored(
            manifest,
            text_ids,
            "is_social_norm_bool",
            lambda item_id, strict=strict: text_pass(
                text[item_id]["result"], strict=strict
            ),
        )

    ensemble_ids = sorted(set(qwen) & set(glm) & set(text))
    for visual in ("glm", "either", "both"):
        for strict_text in (False, True):
            name = f"{visual}_strict_usable_and_text_{'strict' if strict_text else 'basic'}"

            def ensemble_predict(
                item_id: str,
                *,
                visual: str = visual,
                strict_text: bool = strict_text,
            ) -> bool:
                q = strict_v5(qwen[item_id]["result"])
                g = strict_v5(glm[item_id]["result"])
                visual_pass = (
                    g if visual == "glm" else ((q or g) if visual == "either" else (q and g))
                )
                return visual_pass and text_pass(
                    text[item_id]["result"], strict=strict_text
                )

            rules[name] = scored(
                manifest,
                ensemble_ids,
                "gold_usable",
                ensemble_predict,
            )

    feature_names = {
        "motion": "low_level.motion_mean",
        "histogram_change": "low_level.histogram_delta_mean",
        "multiple_people_pose": "keypoints.multiple_people_fraction",
        "person_present_pose": "keypoints.person_present_fraction",
        "xclip_dialogue": (
            "xclip.probabilities.1:a dialogue scene between characters inside a "
        ),
        "xclip_dialogue_margin": (
            "xclip.derived.dialogue_minus_direct_address"
        ),
        "clip_roleplay": (
            "clip.mean_probabilities.0:a staged role play showing a social interacti"
        ),
    }
    feature_ids = sorted(features)
    for name, feature in feature_names.items():
        rule = frozen[feature]
        rules[f"frozen_{name}"] = scored(
            manifest,
            feature_ids,
            "gold_social_scene_visible",
            lambda item_id, feature=feature, rule=rule: apply_frozen_threshold(
                features[item_id][feature], rule
            ),
        )
        rules[f"frozen_{name}"]["feature"] = feature
        rules[f"frozen_{name}"]["threshold"] = rule["threshold"]
        rules[f"frozen_{name}"]["direction"] = rule["direction"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "items": report["items"],
        "valid": report["valid"],
        "summary": {
            name: {
                key: value
                for key, value in rule.items()
                if key in {"precision", "recall", "specificity", "f1", "n"}
            }
            for name, rule in rules.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
