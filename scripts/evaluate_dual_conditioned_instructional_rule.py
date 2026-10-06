#!/usr/bin/env python3
"""Evaluate the frozen dual-conditioned instructional shadow rule.

This script is read-only with respect to the corpus. It joins a sealed manual
manifest to append-only Qwen, GLM, and text-model outputs and writes one
reproducible report plus a JSONL queue containing every selected item.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


STRICT_VISUAL_FIELDS = (
    "social_norm_domain",
    "behavior_occurs_in_scene",
    "affected_party_or_shared_setting_visible",
    "situated_interaction_complete",
    "usable_demo_after_relabel",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def latest_successes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the latest valid record per item from an append-only ledger."""
    successful: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("result") is not None and not row.get("error"):
            successful[row["item_id"]] = row
    return successful


def strict_visual_pass(result: dict[str, Any]) -> bool:
    return all(result.get(field) == "yes" for field in STRICT_VISUAL_FIELDS)


def text_pass(result: dict[str, Any]) -> bool:
    return result.get("social_norm_candidate") == "yes"


def confusion(
    gold: list[bool], predicted: list[bool]
) -> dict[str, int | float | None]:
    tp = sum(g and p for g, p in zip(gold, predicted))
    tn = sum((not g) and (not p) for g, p in zip(gold, predicted))
    fp = sum((not g) and p for g, p in zip(gold, predicted))
    fn = sum(g and (not p) for g, p in zip(gold, predicted))

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "n": len(gold),
        "selected": tp + fp,
        "gold_positive": tp + fn,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "accuracy": ratio(tp + tn, len(gold)),
    }


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def evaluate(
    manifest_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    glm_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
    amendment_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    qwen = latest_successes(qwen_rows)
    glm = latest_successes(glm_rows)
    text = latest_successes(text_rows)
    amendments = {
        int(row["audit_index"]): bool(row["amended_gold_usable"])
        for row in amendment_rows
        if "amended_gold_usable" in row
    }

    joined: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for item in manifest_rows:
        item_id = item["item_id"]
        if item_id not in qwen or item_id not in glm or item_id not in text:
            continue
        q_pass = strict_visual_pass(qwen[item_id]["result"])
        g_pass = strict_visual_pass(glm[item_id]["result"])
        t_pass = text_pass(text[item_id]["result"])
        prediction = q_pass and g_pass and t_pass
        audit_index = int(item["audit_index"])
        original_gold = bool(item["gold_usable"])
        amended_gold = amendments.get(audit_index, original_gold)
        record = {
            "audit_index": audit_index,
            "item_id": item_id,
            "uid": item["uid"],
            "proxy_clip": item.get("proxy_clip"),
            "norm": item.get("norm"),
            "polarity": item.get("polarity"),
            "category": item.get("category"),
            "qwen_strict_visual_pass": q_pass,
            "glm_strict_visual_pass": g_pass,
            "text_social_pass": t_pass,
            "selected": prediction,
            "original_gold_usable": original_gold,
            "amended_gold_usable": amended_gold,
            "qwen_result": qwen[item_id]["result"],
            "glm_result": glm[item_id]["result"],
            "text_result": text[item_id]["result"],
        }
        joined.append(record)
        if prediction:
            selected.append(record)

    predictions = [row["selected"] for row in joined]
    original_metrics = confusion(
        [row["original_gold_usable"] for row in joined], predictions
    )
    amended_metrics = confusion(
        [row["amended_gold_usable"] for row in joined], predictions
    )
    for result in (original_metrics, amended_metrics):
        result["precision_wilson_95"] = wilson_interval(
            int(result["true_positive"]), int(result["selected"])
        )

    report = {
        "rule": "dual_conditioned_v5_strict_usable_and_text_basic",
        "corpus_action": "none_shadow_only",
        "manifest_items": len(manifest_rows),
        "coverage": {
            "qwen_valid": len(qwen),
            "glm_valid": len(glm),
            "text_valid": len(text),
            "joint_valid": len(joined),
            "joint_missing_item_ids": [
                row["item_id"]
                for row in manifest_rows
                if row["item_id"] not in qwen
                or row["item_id"] not in glm
                or row["item_id"] not in text
            ],
        },
        "frozen_original_gold": original_metrics,
        "append_only_amended_gold": amended_metrics,
        "promotion_requirement": {
            "minimum_selected": 10,
            "minimum_manually_verified_precision": 0.9,
            "all_selected_outputs_must_be_manually_reviewed": True,
        },
        "selected_audit_indices": [
            row["audit_index"] for row in selected
        ],
        "notes": [
            "All invalid or missing component outputs fail closed.",
            "The report does not mutate corpus clips, metadata, or routing.",
            "Model-derived precision is provisional until every selected clip is manually reviewed.",
        ],
    }
    return report, selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--glm", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument(
        "--amendments",
        type=Path,
        nargs="+",
        required=True,
        help="One or more append-only amendment ledgers, applied left to right.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    args = parser.parse_args()

    report, selected = evaluate(
        load_jsonl(args.manifest),
        load_jsonl(args.qwen),
        load_jsonl(args.glm),
        load_jsonl(args.text),
        [
            row
            for amendment_path in args.amendments
            for row in load_jsonl(amendment_path)
        ],
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    write_jsonl(args.selected, selected)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
