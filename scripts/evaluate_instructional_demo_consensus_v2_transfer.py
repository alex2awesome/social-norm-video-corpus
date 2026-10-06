#!/usr/bin/env python3
"""Evaluate the frozen V1+V2 demo consensus on a source-disjoint holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_instructional_demo_consensus_v2 import (
        latest_success,
        metric,
        read_jsonl,
        v1_relaxed,
    )
except ModuleNotFoundError:
    from evaluate_instructional_demo_consensus_v2 import (  # type: ignore[no-redef]
        latest_success,
        metric,
        read_jsonl,
        v1_relaxed,
    )


TRUE_VALUES = {"1", "true", "y", "yes"}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_transfer(
    manifest: list[dict[str, Any]],
    manual: list[dict[str, str]],
    v1_rows: list[dict[str, Any]],
    v2_rows: list[dict[str, Any]],
    output_audit: list[dict[str, str]],
    preregistration: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_by_index = {int(row["audit_index"]): row for row in manifest}
    manual_by_index = {int(row["audit_index"]): row for row in manual}
    if len(manifest_by_index) != len(manifest) or len(manual_by_index) != len(manual):
        raise ValueError("duplicate audit_index")
    if set(manifest_by_index) != set(manual_by_index):
        raise ValueError("manifest/manual coverage mismatch")

    v1 = latest_success(v1_rows)
    v2 = latest_success(v2_rows)
    expected_items = {str(row["item_id"]) for row in manifest}
    if set(v1) != expected_items or set(v2) != expected_items:
        raise ValueError("manifest/V1/V2 successful coverage mismatch")

    audit_by_index = {int(row["audit_index"]): row for row in output_audit}
    audit_complete = (
        set(audit_by_index) == set(manifest_by_index)
        and all(
            truthy(row.get("v1_output_reviewed"))
            and truthy(row.get("v2_output_reviewed"))
            for row in audit_by_index.values()
        )
    )

    evaluated = []
    for index in sorted(manifest_by_index):
        source = manifest_by_index[index]
        item_id = str(source["item_id"])
        gold = manual_by_index[index]["visual_demo"].strip().upper()
        if gold not in {"Y", "N"}:
            raise ValueError(f"audit_index {index}: visual_demo must be Y or N")
        p1 = v1_relaxed(v1[item_id]["result"])
        p2 = bool(v2[item_id]["result"]["demo_candidate"])
        consensus = p1 and p2
        evaluated.append({
            "audit_index": index,
            "candidate_id": source.get("candidate_id"),
            "item_id": item_id,
            "uid": source.get("uid"),
            "manual_visual_demo": gold == "Y",
            "manual_visual_form": manual_by_index[index].get("visual_form"),
            "manual_note": manual_by_index[index].get("note"),
            "v1_relaxed": p1,
            "v2_demo_candidate": p2,
            "v1_v2_consensus": consensus,
            "consensus_error": (
                "false_positive" if consensus and gold == "N"
                else "false_negative" if not consensus and gold == "Y"
                else None
            ),
            "v1_result": v1[item_id]["result"],
            "v2_result": v2[item_id]["result"],
            "model_outputs_manually_reviewed": index in audit_by_index and truthy(
                audit_by_index[index].get("v1_output_reviewed")
            ) and truthy(audit_by_index[index].get("v2_output_reviewed")),
            "model_output_review_note": audit_by_index.get(index, {}).get("note"),
        })

    truth = [row["manual_visual_demo"] for row in evaluated]
    metrics = {
        "v1_relaxed": metric(truth, [row["v1_relaxed"] for row in evaluated]),
        "v2": metric(truth, [row["v2_demo_candidate"] for row in evaluated]),
        "v1_v2_consensus": metric(
            truth, [row["v1_v2_consensus"] for row in evaluated]
        ),
    }
    gate = preregistration["fresh_validation_gate"]
    consensus_metric = metrics["v1_v2_consensus"]
    checks = {
        "minimum_clips": len(evaluated) >= int(gate["minimum_clips"]),
        "manual_visual_coverage": len(evaluated) == int(
            gate["required_manual_coverage"]
        ),
        "manual_model_output_audit": audit_complete,
        "minimum_precision": (
            consensus_metric["precision"] is not None
            and consensus_metric["precision"] >= float(gate["minimum_precision"])
        ),
        "minimum_recall": (
            consensus_metric["recall"] is not None
            and consensus_metric["recall"] >= float(gate["minimum_recall"])
        ),
        "minimum_selected": consensus_metric["selected"] >= int(
            gate["minimum_selected"]
        ),
        "source_disjoint_preregistered": bool(
            gate["source_disjoint_from_all_prior_manual_uids"]
        ),
    }
    passed = all(checks.values())
    return evaluated, {
        "kind": "instructional_demo_consensus_v2_fresh_transfer_evaluation",
        "items": len(evaluated),
        "manual_visual_demos": sum(truth),
        "manual_visual_demo_rate": sum(truth) / len(truth),
        "metrics": metrics,
        "checks": checks,
        "preregistered_fresh_gate_passed": passed,
        "status": (
            "passed_fresh_gate_requires_post_scale_transfer_audit"
            if passed else "failed_fresh_validation"
        ),
        "automatic_acceptance": False,
        "acceptance_or_rejection_authority": False,
        "corpus_mutation_authorized": False,
        "all_media_retained": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manual-ledger", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--manual-output-audit", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = evaluate_transfer(
        read_jsonl(args.manifest),
        read_tsv(args.manual_ledger),
        read_jsonl(args.v1),
        read_jsonl(args.v2),
        read_tsv(args.manual_output_audit),
        json.loads(args.preregistration.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary["artifact_sha256"] = {
        "manifest": sha256(args.manifest),
        "manual_ledger": sha256(args.manual_ledger),
        "v1": sha256(args.v1),
        "v2": sha256(args.v2),
        "manual_output_audit": sha256(args.manual_output_audit),
        "preregistration": sha256(args.preregistration),
        "evaluated": sha256(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
