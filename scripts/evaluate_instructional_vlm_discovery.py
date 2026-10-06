#!/usr/bin/env python3
"""Evaluate discovery-stage instructional VLM gates on the frozen 120-item audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def v6_strict(result: dict[str, Any] | None) -> bool:
    return bool(
        result
        and result.get("visually_observable_event") == "yes"
        and result.get("metadata_needed_to_identify_action") == "no"
        and result.get("presentation_or_context_only") == "no"
        and isinstance(result.get("event_start_percent"), (int, float))
        and isinstance(result.get("event_end_percent"), (int, float))
        and result["event_start_percent"] >= 0
        and result["event_end_percent"] >= result["event_start_percent"]
    )


def v8_strict(result: dict[str, Any] | None) -> bool:
    yes_fields = (
        "on_screen_social_event",
        "actor_performs_target_behavior",
        "affected_party_or_shared_context_same_event",
        "behavior_socially_evaluable_from_clip",
        "event_temporally_localized",
        "informal_social_conduct_not_formal_procedure",
        "usable_demo_after_relabel",
    )
    return bool(
        result
        and all(result.get(field) == "yes" for field in yes_fields)
        and result.get("evidence_source")
        in {
            "physical_action",
            "situated_dialogue_or_subtitles",
            "static_depicted_action",
        }
        and result.get("visual_role") == "demonstrated_event"
    )


def load_gold(
    candidate_ledger: Path, control_decisions: Path
) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(candidate_ledger):
        gold[row["item_id"]] = {
            "positive": bool(row["manual_positive"]),
            "audit_band": row["audit_band"],
        }
    for row in read_jsonl(control_decisions):
        if row["item_id"] in gold:
            raise ValueError(f"duplicate gold item {row['item_id']}")
        gold[row["item_id"]] = {
            "positive": row["visual_demo_usable_after_relabel"] == "yes",
            "audit_band": "qwen_reject_control",
        }
    if len(gold) != 120:
        raise ValueError(f"expected 120 gold items, found {len(gold)}")
    return gold


def load_predictions(path: Path) -> dict[str, dict[str, Any] | None]:
    predictions: dict[str, dict[str, Any] | None] = {}
    for row in read_jsonl(path):
        item_id = row["item_id"]
        if item_id in predictions and predictions[item_id] is not None:
            continue
        predictions[item_id] = None if row.get("error") else row.get("result")
    return predictions


def confusion(
    gold: dict[str, dict[str, Any]],
    predicted: dict[str, bool],
    item_ids: list[str],
) -> dict[str, Any]:
    tp = sum(gold[item]["positive"] and predicted[item] for item in item_ids)
    fp = sum(not gold[item]["positive"] and predicted[item] for item in item_ids)
    fn = sum(gold[item]["positive"] and not predicted[item] for item in item_ids)
    tn = sum(not gold[item]["positive"] and not predicted[item] for item in item_ids)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "n": len(item_ids),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "selected": tp + fp,
        "false_positive_item_ids": [
            item
            for item in item_ids
            if not gold[item]["positive"] and predicted[item]
        ],
        "false_negative_item_ids": [
            item
            for item in item_ids
            if gold[item]["positive"] and not predicted[item]
        ],
    }


def evaluate(
    candidate_ledger: Path,
    control_decisions: Path,
    qwen: Path,
    glm: Path,
    rubric: str,
) -> dict[str, Any]:
    gate: Callable[[dict[str, Any] | None], bool] = {
        "v6": v6_strict,
        "v8": v8_strict,
    }[rubric]
    gold = load_gold(candidate_ledger, control_decisions)
    qwen_rows = load_predictions(qwen)
    glm_rows = load_predictions(glm)
    if set(qwen_rows) != set(gold):
        raise ValueError("Qwen prediction IDs do not match gold")
    if set(glm_rows) != set(gold):
        raise ValueError("GLM prediction IDs do not match gold")
    predictions = {
        "qwen": {item: gate(qwen_rows[item]) for item in gold},
        "glm": {item: gate(glm_rows[item]) for item in gold},
    }
    predictions["conjunction"] = {
        item: predictions["qwen"][item] and predictions["glm"][item]
        for item in gold
    }
    bands = {
        "all_stratified": list(gold),
        "dual_strict": [
            item for item, row in gold.items() if row["audit_band"] == "dual_strict"
        ],
        "qwen_strict_glm_reject": [
            item
            for item, row in gold.items()
            if row["audit_band"] == "qwen_strict_glm_reject"
        ],
        "qwen_reject_control": [
            item
            for item, row in gold.items()
            if row["audit_band"] == "qwen_reject_control"
        ],
    }
    return {
        "kind": "instructional_vlm_gate_discovery",
        "rubric": rubric,
        "artifacts": {
            "candidate_ledger_sha256": sha256(candidate_ledger),
            "control_decisions_sha256": sha256(control_decisions),
            "qwen_sha256": sha256(qwen),
            "glm_sha256": sha256(glm),
        },
        "metrics": {
            model: {
                band: confusion(gold, prediction, item_ids)
                for band, item_ids in bands.items()
            }
            for model, prediction in predictions.items()
        },
        "warning": (
            "This cohort was used for mechanism discovery and is stratified; "
            "these metrics are not a fresh validation or corpus prevalence estimate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-ledger", required=True, type=Path)
    parser.add_argument("--control-decisions", required=True, type=Path)
    parser.add_argument("--qwen", required=True, type=Path)
    parser.add_argument("--glm", required=True, type=Path)
    parser.add_argument("--rubric", choices=("v6", "v8"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.candidate_ledger,
        args.control_decisions,
        args.qwen,
        args.glm,
        args.rubric,
    )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                model: metrics["all_stratified"]
                for model, metrics in result["metrics"].items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
