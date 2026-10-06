#!/usr/bin/env python3
"""Apply the frozen scale-transfer gate to witnessed video+ASR retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    prereg: dict[str, Any],
    clip_report: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    change_rows: list[dict[str, str]],
) -> dict[str, Any]:
    gate = prereg["required_gate"]
    metrics = clip_report["rules"]["strict_bystander_reaction"]["fail_closed"]
    manifest_ids = {row["candidate_id"] for row in manifest_rows}
    successful_ids = {
        row["candidate_id"] for row in model_rows if row.get("error") is None
    }
    representation_valid = bool(manifest_rows) and all(
        row.get("two_stage_full_proxy_then_candidate_cut") is True
        and row.get("max_source_frames") == 96
        and row.get("max_width") == 512
        for row in manifest_rows
    )
    coverage_complete = successful_ids == manifest_ids
    pass_precision = metrics["precision"] is not None and metrics["precision"] >= gate["minimum_precision"]
    pass_recall = metrics["recall"] is not None and metrics["recall"] >= gate["minimum_recall"]
    pass_size = clip_report["manual_clips"] >= gate["minimum_manual_clips"]
    passed = all((representation_valid, coverage_complete, pass_precision, pass_recall, pass_size))
    outcomes: dict[str, int] = {}
    for row in change_rows:
        for value in row.get("change_audit", "").split(","):
            if ":" not in value:
                continue
            outcome = value.split(":", 1)[1]
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return {
        "kind": "witnessed_video_asr_scale_transfer_gate_v1",
        "status": (
            "passed_for_shadow_candidate_generation_and_review_ranking"
            if passed else "failed_transfer_do_not_scale"
        ),
        "passed": passed,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
        "target": "clip_level_strict_bystander_reaction_retrieval_not_social_norm_or_keep_label",
        "representation": "silent_full_clip_proxy_max96_frames_512px_then_candidate_cut_plus_asr_context",
        "manual_clips": clip_report["manual_clips"],
        "candidate_windows": len(manifest_rows),
        "successful_model_windows": len(successful_ids),
        "metrics": metrics,
        "gate_checks": {
            "representation_valid": representation_valid,
            "coverage_complete": coverage_complete,
            "precision_at_least_minimum": pass_precision,
            "recall_at_least_minimum": pass_recall,
            "manual_clip_count_at_least_minimum": pass_size,
        },
        "decision_change_audit": {
            "changed_candidates": len(change_rows),
            "atom_change_outcomes": dict(sorted(outcomes.items())),
        },
        "policy": "shadow_review_routing_only_no_keep_reject_or_delete",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--clip-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--change-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.change_audit.open(newline="") as handle:
        change_rows = list(csv.DictReader(handle, delimiter="\t"))
    report = evaluate(
        json.loads(args.prereg.read_text()),
        json.loads(args.clip_report.read_text()),
        read_jsonl(args.manifest),
        read_jsonl(args.model),
        change_rows,
    )
    report["artifact_sha256"] = {
        "prereg": sha256(args.prereg),
        "clip_report": sha256(args.clip_report),
        "manifest": sha256(args.manifest),
        "model": sha256(args.model),
        "change_audit": sha256(args.change_audit),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
