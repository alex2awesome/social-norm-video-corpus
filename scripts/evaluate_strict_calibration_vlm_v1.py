#!/usr/bin/env python3
"""Evaluate pillar-specific Qwen judgments against frozen manual calibration."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def instructional_positive(result: dict[str, Any]) -> bool:
    return (
        all(
            result.get(key) == "yes"
            for key in (
                "on_screen_social_event",
                "actor_performs_target_behavior",
                "affected_party_or_shared_context_same_event",
                "behavior_socially_evaluable_from_clip",
                "event_temporally_localized",
                "informal_social_conduct_not_formal_procedure",
                "usable_demo_after_relabel",
            )
        )
        and result.get("evidence_source")
        in {"physical_action", "situated_dialogue_or_subtitles", "static_depicted_action"}
        and result.get("visual_role") == "demonstrated_event"
    )


def commentary_positive(result: dict[str, Any]) -> bool:
    return (
        result.get("social_norm_domain") == "yes"
        and result.get("situated_social_scenario_visible") == "yes"
        and result.get("observable_social_behavior_or_speech") == "yes"
        and result.get("usable_demo_after_relabel") == "yes"
        and result.get("localization_quality") == "clean"
    )


def witnessed_positive(result: dict[str, Any]) -> bool:
    end = result.get("action_end_percent", -1)
    start = result.get("reaction_start_percent", -1)
    return (
        result.get("action_visible") == "yes"
        and result.get("action_voluntary") == "yes"
        and result.get("expectation_kind")
        in {"interpersonal_treatment", "shared_public_conduct"}
        and result.get("reaction_visible_or_audibly_grounded") == "yes"
        and result.get("reaction_source_role")
        in {"bystander", "authority_or_host", "organic_audience"}
        and result.get("reaction_content")
        in {"targeted_objection", "correction_or_sanction", "protective_intervention"}
        and result.get("action_established_before_reaction") == "yes"
        and result.get("reaction_targets_action") == "yes"
        and result.get("authenticity") in {"organic", "hidden_camera_genuine"}
        and result.get("pre_reaction_demo_quality")
        in {"clear_visual", "clear_audiovisual"}
        and isinstance(end, (int, float))
        and isinstance(start, (int, float))
        and 0 <= end < start <= 100
    )


PREDICATES = {
    "instructional": instructional_positive,
    "witnessed": witnessed_positive,
    "commentary": commentary_positive,
}


def metrics(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    tp = sum(gold and pred for gold, pred in pairs)
    fp = sum(not gold and pred for gold, pred in pairs)
    fn = sum(gold and not pred for gold, pred in pairs)
    tn = sum(not gold and not pred for gold, pred in pairs)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def evaluate(manifest: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    gold = {row["item_id"]: row for row in manifest}
    predictions = [row for row in predictions if row.get("error") is None]
    if len({row["item_id"] for row in predictions}) != len(predictions):
        raise ValueError("duplicate successful model response")
    if set(gold) != {row["item_id"] for row in predictions}:
        raise ValueError("gold/prediction item sets differ")
    by_pillar: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    decisions = []
    for row in sorted(predictions, key=lambda value: gold[value["item_id"]]["audit_index"]):
        target = gold[row["item_id"]]
        pillar = target["pillar"]
        scene_salvage = bool(target["gold_social_scene_visible"])
        pillar_usable = (
            target["gold_strict_decision"] == "accept"
            or (
                pillar != "witnessed"
                and target["gold_strict_decision"] == "accept_with_repairs"
            )
        )
        prediction = PREDICATES[pillar](row["result"])
        by_pillar[pillar].append((pillar_usable, prediction))
        decisions.append(
            {
                "audit_index": target["audit_index"],
                "item_id": row["item_id"],
                "pillar": pillar,
                "gold_scene_salvage": scene_salvage,
                "gold_pillar_usable": pillar_usable,
                "gold_strict_decision": target["gold_strict_decision"],
                "model_positive": prediction,
                "agreement": pillar_usable == prediction,
            }
        )
    return {
        "items": len(decisions),
        "target": "human_pillar_usable_including_non_witnessed_repairs",
        "by_pillar": {pillar: metrics(pairs) for pillar, pairs in sorted(by_pillar.items())},
        "disagreements": [row for row in decisions if not row["agreement"]],
        "decisions": decisions,
        "automatic_promotion": False,
        "corpus_mutated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predictions = []
    for path in args.prediction:
        predictions.extend(load_jsonl(path))
    result = evaluate(load_jsonl(args.manifest), predictions)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["by_pillar"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
