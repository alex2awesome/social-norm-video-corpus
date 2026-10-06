#!/usr/bin/env python3
"""Apply the frozen blind-episode gate to a fresh, manually audited cohort."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def assess_gate(
    evaluation: dict[str, Any],
    preregistration: dict[str, Any],
    output_audit_rows: list[dict[str, str]],
) -> dict[str, Any]:
    gate = preregistration["fresh_validation_gate"]
    metric = evaluation["episode_without_completeness_development"]["qwen"]
    required_audit = int(gate["required_manual_coverage"])
    output_audit_complete = (
        len(output_audit_rows) == required_audit
        and all(
            row.get("episode_output_review")
            and row.get("symbolic_output_review")
            for row in output_audit_rows
        )
    )
    checks = {
        "minimum_clips": evaluation["items"] >= int(gate["minimum_clips"]),
        "manual_visual_coverage": (
            evaluation.get("manual_coverage_complete") is True
            and evaluation["items"] == required_audit
        ),
        "manual_model_output_audit": output_audit_complete,
        "minimum_precision": (
            metric["precision"] is not None
            and metric["precision"] >= float(gate["minimum_precision"])
        ),
        "minimum_recall": (
            metric["recall"] is not None
            and metric["recall"] >= float(gate["minimum_recall"])
        ),
        "minimum_selected": metric["selected"] >= int(gate["minimum_selected"]),
    }
    passed = all(checks.values())
    return {
        "kind": "instructional_blind_episode_relaxed_transfer_validation_v1",
        "target": gate["target"],
        "frozen_model": preregistration["frozen_rule"]["model"],
        "frozen_prompt": preregistration["frozen_rule"]["prompt"],
        "items": evaluation["items"],
        "manual_visual_demos": evaluation["manual_visual_demos"],
        "manual_exact_usable": evaluation["manual_usable_demos"],
        "metric": metric,
        "checks": checks,
        "preregistered_pass": passed,
        "status": (
            "passed_for_shadow_candidate_routing_only"
            if passed else "failed_fresh_validation"
        ),
        "automatic_acceptance": False,
        "acceptance_or_rejection_authority": False,
        "corpus_mutation_authorized": False,
        "all_media_retained": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--manual-output-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    preregistration = json.loads(args.preregistration.read_text())
    with args.manual_output_audit.open() as handle:
        audit_rows = list(csv.DictReader(handle, delimiter="\t"))
    result = assess_gate(evaluation, preregistration, audit_rows)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
