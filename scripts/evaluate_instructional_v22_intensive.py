#!/usr/bin/env python3
"""Evaluate the two-stage instructional V22 pipeline on a complete manual audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_success(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        if row.get("error") in (None, "") and isinstance(row.get("result"), dict):
            output[str(row["item_id"])] = row
    return output


def model_records(
    rows: list[dict[str, Any]],
    decision_field: str,
    error_policy: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    successes = latest_success(rows)
    attempts = {str(row["item_id"]): row for row in rows}
    failed_items = sorted(set(attempts) - set(successes))
    if error_policy == "fail_closed":
        for item_id in failed_items:
            attempt = attempts[item_id]
            successes[item_id] = {
                **attempt,
                "result": {decision_field: "no"},
                "evaluation_fallback": "model_error_as_negative",
            }
    elif error_policy != "strict":
        raise ValueError(f"unsupported model error policy: {error_policy}")
    return successes, failed_items


def as_yes(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"yes", "no", "y", "n"}:
        raise ValueError(f"expected yes/no manual label, got {value!r}")
    return normalized in {"yes", "y"}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def metric(labels: list[bool], predictions: list[bool]) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("label/prediction length mismatch")
    tp = sum(y and p for y, p in zip(labels, predictions))
    fp = sum(not y and p for y, p in zip(labels, predictions))
    fn = sum(y and not p for y, p in zip(labels, predictions))
    tn = sum(not y and not p for y, p in zip(labels, predictions))
    selected = tp + fp
    positives = tp + fn
    return {
        "n": len(labels),
        "selected": selected,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / selected if selected else None,
        "precision_wilson_95": wilson(tp, selected),
        "recall": tp / positives if positives else None,
        "recall_wilson_95": wilson(tp, positives),
    }


def evaluate(
    blind_manifest: list[dict[str, Any]],
    semantic_manifest: list[dict[str, Any]],
    ledger: list[dict[str, str]],
    visual_outputs: dict[str, list[dict[str, Any]]],
    semantic_outputs: dict[str, list[dict[str, Any]]],
    expected_count: int,
    model_error_policy: str = "strict",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not visual_outputs or not semantic_outputs:
        raise ValueError("visual and semantic model outputs are both required")
    for name, rows in {
        "blind manifest": blind_manifest,
        "semantic manifest": semantic_manifest,
        "manual ledger": ledger,
    }.items():
        if len(rows) != expected_count:
            raise ValueError(f"expected {expected_count} rows in {name}, got {len(rows)}")
    semantic_by_item = {str(row["item_id"]): row for row in semantic_manifest}
    ledger_by_item = {str(row["item_id"]): row for row in ledger}
    if len(semantic_by_item) != expected_count or len(ledger_by_item) != expected_count:
        raise ValueError("duplicate semantic or manual item_id")
    visual_pairs = {
        name: model_records(rows, "demo_pass", model_error_policy)
        for name, rows in visual_outputs.items()
    }
    semantic_pairs = {
        name: model_records(rows, "semantic_pass", model_error_policy)
        for name, rows in semantic_outputs.items()
    }
    visuals = {name: pair[0] for name, pair in visual_pairs.items()}
    semantics = {name: pair[0] for name, pair in semantic_pairs.items()}
    model_failed_items = {
        "visual": {name: pair[1] for name, pair in visual_pairs.items()},
        "semantic": {name: pair[1] for name, pair in semantic_pairs.items()},
    }
    evaluated = []
    for blind in blind_manifest:
        item_id = str(blind["item_id"])
        if item_id not in semantic_by_item or item_id not in ledger_by_item:
            raise ValueError(f"missing joined row: {item_id}")
        source = semantic_by_item[item_id]
        manual = ledger_by_item[item_id]
        if str(manual["candidate_id"]) != str(blind["candidate_id"]):
            raise ValueError(f"candidate mismatch: {item_id}")
        visual_decisions = {}
        semantic_decisions = {}
        for name, outputs in visuals.items():
            if item_id not in outputs:
                raise ValueError(f"missing visual {name} output: {item_id}")
            visual_decisions[name] = outputs[item_id]["result"].get("demo_pass") == "yes"
        for name, outputs in semantics.items():
            if item_id not in outputs:
                raise ValueError(f"missing semantic {name} output: {item_id}")
            semantic_decisions[name] = outputs[item_id]["result"].get("semantic_pass") == "yes"
        visual_consensus = all(visual_decisions.values())
        semantic_consensus = all(semantic_decisions.values())
        evaluated.append(
            {
                **blind,
                "source_platform": source.get("source_platform") or str(source.get("uid", "")).split("__", 1)[0],
                "category": source.get("category") or "unknown",
                "polarity": source.get("polarity") or "unknown",
                "manual_visual_form": manual["visual_form"],
                "manual_visual_demo": as_yes(manual["visual_demo"]),
                "manual_semantic_alignment": as_yes(manual["semantic_alignment"]),
                "manual_complete_demo": as_yes(manual["complete_demo"]),
                "manual_usable_demo": as_yes(manual["usable_demo"]),
                "manual_note": manual.get("note", ""),
                "visual_decisions": visual_decisions,
                "semantic_decisions": semantic_decisions,
                "visual_consensus": visual_consensus,
                "semantic_consensus": semantic_consensus,
                "pipeline_consensus": visual_consensus and semantic_consensus,
            }
        )
    visual_gold = [row["manual_visual_demo"] for row in evaluated]
    usable_gold = [row["manual_usable_demo"] for row in evaluated]
    visual_rules = {
        name: metric(visual_gold, [row["visual_decisions"][name] for row in evaluated])
        for name in visuals
    }
    visual_rules["all_model_consensus"] = metric(
        visual_gold, [row["visual_consensus"] for row in evaluated]
    )
    semantic_rules = {
        name: metric(usable_gold, [row["semantic_decisions"][name] for row in evaluated])
        for name in semantics
    }
    semantic_rules["all_model_consensus"] = metric(
        usable_gold, [row["semantic_consensus"] for row in evaluated]
    )
    pipeline = metric(usable_gold, [row["pipeline_consensus"] for row in evaluated])

    slices: dict[str, dict[str, Any]] = {}
    for field in ("manual_visual_form", "source_platform", "category", "polarity"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in evaluated:
            groups[str(row[field])].append(row)
        slices[field] = {
            value: metric(
                [row["manual_usable_demo"] for row in rows],
                [row["pipeline_consensus"] for row in rows],
            )
            for value, rows in sorted(groups.items())
        }
    selected_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        if row["pipeline_consensus"]:
            selected_by_uid[str(row["uid"])].append(row)
    exact_good_uids = sum(
        all(row["manual_usable_demo"] for row in rows)
        for rows in selected_by_uid.values()
    )
    source_cluster = {
        "selected_sources": len(selected_by_uid),
        "exact_good_sources": exact_good_uids,
        "precision": exact_good_uids / len(selected_by_uid) if selected_by_uid else None,
        "precision_wilson_95": wilson(exact_good_uids, len(selected_by_uid)),
    }
    summary = {
        "kind": "instructional_v22_two_stage_intensive_manual_evaluation",
        "policy": "read_only_shadow_no_keep_reject_or_corpus_mutation",
        "items": len(evaluated),
        "manual_coverage_complete": True,
        "model_error_policy": model_error_policy,
        "model_failed_items": model_failed_items,
        "manual_visual_demos": sum(visual_gold),
        "manual_usable_demos": sum(usable_gold),
        "visual_stage": visual_rules,
        "semantic_stage": semantic_rules,
        "pipeline_consensus": pipeline,
        "source_cluster_exact": source_cluster,
        "slices": slices,
    }
    return evaluated, summary


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--visual-output", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--semantic-output", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument(
        "--model-error-policy",
        choices=("strict", "fail_closed"),
        default="strict",
        help="In fail_closed mode, an attempted item with no successful parse votes no.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    visual_paths = {name: Path(path) for name, path in args.visual_output}
    semantic_paths = {name: Path(path) for name, path in args.semantic_output}
    rows, summary = evaluate(
        read_jsonl(args.blind_manifest),
        read_jsonl(args.semantic_manifest),
        read_tsv(args.ledger),
        {name: read_jsonl(path) for name, path in visual_paths.items()},
        {name: read_jsonl(path) for name, path in semantic_paths.items()},
        args.expected_count,
        args.model_error_policy,
    )
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary["artifact_sha256"] = {
        "blind_manifest": sha256(args.blind_manifest),
        "semantic_manifest": sha256(args.semantic_manifest),
        "ledger": sha256(args.ledger),
        "visual_outputs": {name: sha256(path) for name, path in visual_paths.items()},
        "semantic_outputs": {name: sha256(path) for name, path in semantic_paths.items()},
        "evaluated": sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
