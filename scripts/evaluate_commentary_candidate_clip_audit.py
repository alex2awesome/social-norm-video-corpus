#!/usr/bin/env python3
"""Validate and summarize the 32-item transformed commentary clip audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PASS = {"pass_visual_clean", "pass_instructional_reroute_clean"}


def evaluate(
    plan_path: Path, manifest_path: Path, failures_path: Path, ledger_path: Path
) -> dict:
    plan = [json.loads(line) for line in plan_path.read_text().splitlines() if line]
    manifest = [
        json.loads(line) for line in manifest_path.read_text().splitlines() if line
    ]
    failures = [
        json.loads(line) for line in failures_path.read_text().splitlines() if line
    ]
    with ledger_path.open(newline="") as handle:
        ledger = list(csv.DictReader(handle, delimiter="\t"))
    plan_indices = {int(row["audit_index"]) for row in plan}
    artifact_indices = {int(row["audit_index"]) for row in manifest + failures}
    ledger_indices = {int(row["audit_index"]) for row in ledger}
    if len(plan) != 32 or len(ledger) != 32:
        raise ValueError("plan and manual ledger must each contain 32 items")
    if plan_indices != artifact_indices or plan_indices != ledger_indices:
        raise ValueError("plan, artifacts, and manual ledger cover different indices")
    if len(manifest) != 30 or len(failures) != 2:
        raise ValueError("expected 30 rendered artifacts and two explicit failures")
    by_index = {int(row["audit_index"]): row for row in ledger}
    for row in failures:
        if by_index[int(row["audit_index"])]["outcome"] != "render_failed":
            raise ValueError("render failure is not manually recorded")
    outcomes = Counter(row["outcome"] for row in ledger)
    routes = Counter(row["final_route"] for row in ledger)
    passed = sum(outcomes[name] for name in PASS)
    repair = sum(count for name, count in outcomes.items() if name.startswith("repair_"))
    return {
        "kind": "commentary_dual_exact_candidate_post_transform_manual_audit",
        "policy": "shadow_only_no_corpus_mutation",
        "planned": len(plan),
        "rendered": len(manifest),
        "render_failed": len(failures),
        "passed_first_transform": passed,
        "passed_first_transform_rate": passed / len(plan),
        "clean_commentary_candidates": outcomes["pass_visual_clean"],
        "clean_instructional_reroutes": outcomes[
            "pass_instructional_reroute_clean"
        ],
        "repair_required": repair,
        "source_false_positive": outcomes["fail_source_event_absent"],
        "outcome_counts": dict(sorted(outcomes.items())),
        "route_counts": dict(sorted(routes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.plan, args.manifest, args.failures, args.ledger)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
