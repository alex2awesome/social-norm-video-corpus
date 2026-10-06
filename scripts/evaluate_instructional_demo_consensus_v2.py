#!/usr/bin/env python3
"""Evaluate V1/V2 instructional-demo consensus on frozen manual cohorts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_success(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        if row.get("error") in (None, "") and isinstance(row.get("result"), dict):
            result[str(row["item_id"])] = row
    return result


def v1_relaxed(result: dict[str, Any]) -> bool:
    return (
        result["visual_context"] in {"connected_episode", "roleplay_episode"}
        and result["participant_grounding"]
        in {"affected_party_present", "shared_audience_present"}
        and (
            result["physical_social_action"] == "yes"
            or result["quote_function"] == "situated_to_present_party"
        )
    )


def wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def metric(gold: list[bool], predictions: list[bool]) -> dict[str, Any]:
    if len(gold) != len(predictions) or not gold:
        raise ValueError("gold and predictions require equal non-empty coverage")
    tp = sum(y and p for y, p in zip(gold, predictions))
    fp = sum(not y and p for y, p in zip(gold, predictions))
    fn = sum(y and not p for y, p in zip(gold, predictions))
    tn = sum(not y and not p for y, p in zip(gold, predictions))
    selected = tp + fp
    positives = tp + fn
    precision = tp / selected if selected else None
    recall = tp / positives if positives else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "selected": selected, "manual_positive": positives,
        "precision": precision, "recall": recall,
        "precision_wilson_95": wilson(tp, selected),
        "recall_wilson_95": wilson(tp, positives),
    }


def evaluate_rows(
    cohort: str,
    gold: dict[str, tuple[bool, str]],
    v1: dict[str, dict[str, Any]],
    v2: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(gold) != set(v1) or set(gold) != set(v2):
        raise ValueError(f"{cohort}: gold/V1/V2 coverage mismatch")
    rows = []
    for item_id, (manual, note) in gold.items():
        v1_result = v1[item_id]
        v2_result = v2[item_id]["result"]
        p1 = v1_relaxed(v1_result)
        p2 = bool(v2_result["demo_candidate"])
        consensus = p1 and p2
        rows.append({
            "cohort": cohort,
            "item_id": item_id,
            "manual_visual_demo": manual,
            "manual_note": note,
            "v1_relaxed": p1,
            "v2_demo_candidate": p2,
            "v1_v2_consensus": consensus,
            "consensus_error": (
                "false_positive" if consensus and not manual
                else "false_negative" if manual and not consensus
                else None
            ),
            "v1_result": v1_result,
            "v2_result": v2_result,
            "manual_output_reviewed": True,
        })
    truth = [row["manual_visual_demo"] for row in rows]
    return rows, {
        "items": len(rows),
        "manual_visual_demos": sum(truth),
        "v1_relaxed": metric(truth, [row["v1_relaxed"] for row in rows]),
        "v2": metric(truth, [row["v2_demo_candidate"] for row in rows]),
        "v1_v2_consensus": metric(truth, [row["v1_v2_consensus"] for row in rows]),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-evaluated", type=Path, required=True)
    parser.add_argument("--fresh-v2", type=Path, required=True)
    parser.add_argument("--scale-ledger", type=Path, required=True)
    parser.add_argument("--scale-v1", type=Path, required=True)
    parser.add_argument("--scale-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manual-ledger", type=Path, required=True)
    args = parser.parse_args()

    fresh_source = read_jsonl(args.fresh_evaluated)
    fresh_gold = {
        str(row["item_id"]): (
            bool(row["manual_visual_demo"]), str(row.get("manual_note") or ""),
        )
        for row in fresh_source
    }
    fresh_v1 = {
        str(row["item_id"]): row["episode_results"]["qwen"]
        for row in fresh_source
    }
    fresh_v2 = latest_success(read_jsonl(args.fresh_v2))

    scale_manual = read_tsv(args.scale_ledger)
    scale_gold = {
        "instructional:dailymotion__" + row["item_id"]: (
            row["manual_visual_demo"] == "true", row["note"],
        )
        for row in scale_manual
    }
    scale_v1_rows = {
        str(row["item_id"]): row["atomic_result"]
        for row in read_jsonl(args.scale_v1)
        if str(row["item_id"]) in scale_gold and isinstance(row.get("atomic_result"), dict)
    }
    scale_v2 = latest_success(read_jsonl(args.scale_v2))

    fresh_rows, fresh_summary = evaluate_rows(
        "fresh100", fresh_gold, fresh_v1,
        {key: fresh_v2[key] for key in fresh_gold if key in fresh_v2},
    )
    scale_rows, scale_summary = evaluate_rows(
        "scale_spot30", scale_gold, scale_v1_rows,
        {key: scale_v2[key] for key in scale_gold if key in scale_v2},
    )
    evaluated = fresh_rows + scale_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evaluated)
    )
    with args.manual_ledger.open("w", newline="") as handle:
        fieldnames = [
            "cohort", "item_id", "manual_visual_demo", "v1_relaxed",
            "v2_demo_candidate", "v1_v2_consensus", "consensus_error",
            "manual_output_reviewed", "manual_note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in evaluated:
            writer.writerow({key: row.get(key) for key in fieldnames})

    fresh_consensus = fresh_summary["v1_v2_consensus"]
    scale_consensus = scale_summary["v1_v2_consensus"]
    candidate_for_new_holdout = (
        fresh_consensus["precision"] is not None
        and fresh_consensus["precision"] >= 0.8
        and fresh_consensus["recall"] is not None
        and fresh_consensus["recall"] >= 0.4
        and fresh_consensus["selected"] >= 10
        and scale_consensus["precision"] is not None
        and scale_consensus["precision"] >= 0.8
    )
    summary = {
        "kind": "instructional_demo_consensus_v2_development_evaluation",
        "cohorts": {"fresh100": fresh_summary, "scale_spot30": scale_summary},
        "manual_model_outputs_reviewed": len(evaluated),
        "manual_review_complete": len(evaluated) == 130,
        "candidate_for_new_blinded_holdout": candidate_for_new_holdout,
        "status": "development_only_not_eligible_for_operational_registration",
        "scale_warning": "Scale cohort is prediction-stratified; its recall is diagnostic, not a population estimate.",
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
        "artifact_sha256": {
            "fresh_evaluated": sha256(args.fresh_evaluated),
            "fresh_v2": sha256(args.fresh_v2),
            "scale_ledger": sha256(args.scale_ledger),
            "scale_v1": sha256(args.scale_v1),
            "scale_v2": sha256(args.scale_v2),
            "evaluated": sha256(args.output),
            "manual_ledger": sha256(args.manual_ledger),
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
