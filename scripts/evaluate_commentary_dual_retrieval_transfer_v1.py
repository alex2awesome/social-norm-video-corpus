#!/usr/bin/env python3
"""Evaluate the preregistered source-disjoint commentary dual-VLM transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_commentary_title_event_expansion import latest_success, read_jsonl, read_tsv
    from scripts.evaluate_commentary_title_vlm_full import retrieval_core
else:
    from evaluate_commentary_title_event_expansion import latest_success, read_jsonl, read_tsv
    from evaluate_commentary_title_vlm_full import retrieval_core


TRINARY = {"yes", "no", "uncertain"}
ERROR_MECHANISMS = {
    "none", "invented_action", "actor_target_confusion", "caption_as_event",
    "aftermath_only", "talking_head_or_interview", "tiny_inset_uncertain",
    "missed_visible_event", "bad_bounds", "other", "uncertain",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if not n:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def validate_output_audit(
    rows: list[dict[str, str]], expected: set[tuple[int, str]]
) -> None:
    observed: set[tuple[int, str]] = set()
    for row in rows:
        key = (int(row["audit_index"]), row["model"])
        if key in observed:
            raise ValueError("duplicate manual model-output audit row")
        observed.add(key)
        if row.get("output_visually_supported") not in TRINARY:
            raise ValueError(f"{key}: invalid output_visually_supported")
        if row.get("error_mechanism") not in ERROR_MECHANISMS:
            raise ValueError(f"{key}: invalid error_mechanism")
        if not row.get("manual_evidence", "").strip():
            raise ValueError(f"{key}: missing manual_evidence")
    if observed != expected:
        raise ValueError("manual model-output audit does not exactly cover successful outputs")


def evaluate(
    ledger: list[dict[str, str]], model_rows: dict[str, list[dict[str, Any]]],
    output_audit: list[dict[str, str]], contract: dict[str, Any],
) -> dict[str, Any]:
    manual = {int(row["audit_index"]): row for row in ledger}
    if len(manual) != len(ledger):
        raise ValueError("duplicate manual audit index")
    evaluable = {
        index: row for index, row in manual.items()
        if row.get("render_status") == "ok"
        and row.get("usable_for_visual_localization_review") in TRINARY
    }
    models = {name: latest_success(rows) for name, rows in model_rows.items()}
    expected_models = set(contract["models"])
    if set(models) != expected_models:
        raise ValueError("model set does not match preregistration")
    for name, rows in models.items():
        missing = set(evaluable) - set(rows)
        if missing:
            raise ValueError(f"missing {name} outputs: {sorted(missing)}")
    expected_audit = {(index, model) for index in evaluable for model in expected_models}
    validate_output_audit(output_audit, expected_audit)
    predictions = {
        index: all(retrieval_core(models[name][index]) for name in expected_models)
        for index in evaluable
    }
    gold = {
        index: row["usable_for_visual_localization_review"] == "yes"
        for index, row in evaluable.items()
    }
    selected = {index for index, value in predictions.items() if value}
    positives = {index for index, value in gold.items() if value}
    tp, fp, fn = len(selected & positives), len(selected - positives), len(positives - selected)
    precision = tp / len(selected) if selected else None
    recall = tp / len(positives) if positives else None
    interval = wilson(tp, len(selected))
    gate = contract["gate"]
    checks = {
        "minimum_evaluable_rendered_items": len(evaluable) >= int(gate["minimum_evaluable_rendered_items"]),
        "manual_model_output_audit_complete": len(output_audit) == len(expected_audit),
        "minimum_selected": len(selected) >= int(gate["minimum_selected"]),
        "minimum_precision": precision is not None and precision >= float(gate["minimum_precision"]),
        "minimum_precision_wilson_95_lower": interval is not None and interval[0] >= float(gate["minimum_precision_wilson_95_lower"]),
        "recall_reported": recall is not None,
    }
    passed = all(checks.values())
    return {
        "kind": "commentary_dual_retrieval_core_source_disjoint_transfer_v1",
        "evaluable_rendered_items": len(evaluable),
        "manual_model_outputs_reviewed": len(output_audit),
        "metrics": {
            "selected": len(selected), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "precision_wilson_95": interval, "recall": recall,
        },
        "gate_checks": checks,
        "review_ranking_rule_retained": passed,
        "allowed_use": "review_ranking_only" if passed else "none_disable_rule",
        "automatic_acceptance": False,
        "corpus_mutated": False,
    }


def read_audit(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-output", action="append", nargs=2, metavar=("MODEL", "PATH"), required=True)
    parser.add_argument("--model-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("refusing to overwrite evaluation artifact")
    model_paths = {name: Path(path) for name, path in args.model_output}
    report = evaluate(
        read_tsv(args.ledger),
        {name: read_jsonl(path) for name, path in model_paths.items()},
        read_audit(args.model_audit), json.loads(args.contract.read_text()),
    )
    report["artifact_sha256"] = {
        "ledger": sha256(args.ledger), "contract": sha256(args.contract),
        "model_audit": sha256(args.model_audit),
        "model_outputs": {name: sha256(path) for name, path in model_paths.items()},
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["review_ranking_rule_retained"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
