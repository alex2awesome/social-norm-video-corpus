#!/usr/bin/env python3
"""Apply the preregistered witnessed corpus transfer gate, fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assess(prereg: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    gate = prereg["gate"]
    uniform = evaluation.get("clip_cohort_metrics", {}).get(
        "uniform_probability_sample", {}
    )
    metrics = uniform.get("routes", {}).get(
        "strict_bystander_reaction", {}
    ).get("fail_closed", {})
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    checks = {
        "source_disjoint": evaluation.get("source_disjoint") is True,
        "manual_candidate_review_complete": (
            evaluation.get("manual_candidate_review_complete") is True
        ),
        "model_candidate_coverage": (
            float(evaluation.get("population_model_candidate_coverage") or 0)
            >= float(gate["minimum_model_candidate_coverage"])
        ),
        "minimum_uniform_clips": (
            int(uniform.get("clips") or 0) >= int(gate["minimum_uniform_clips"])
        ),
        "uniform_strict_reaction_precision": (
            precision is not None
            and float(precision) >= float(gate["minimum_uniform_strict_reaction_precision"])
        ),
        "uniform_strict_reaction_recall": (
            recall is not None
            and float(recall) >= float(gate["minimum_uniform_strict_reaction_recall"])
        ),
    }
    passed = all(checks.values())
    return {
        "kind": "witnessed_video_asr_corpus_gate_v1",
        "target": "strict_contemporaneous_third_party_bystander_reaction_retrieval",
        "checks": checks,
        "metrics": metrics,
        "passed": passed,
        "maintenance_action": (
            "retain_shadow_candidate_generation_and_review_ranking"
            if passed else "disable_rule"
        ),
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
        "policy": "audit_gated_shadow_routing_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    prereg = json.loads(args.preregistration.read_text())
    evaluation = json.loads(args.evaluation.read_text())
    report = assess(prereg, evaluation)
    report["artifact_sha256"] = {
        "preregistration": sha256(args.preregistration),
        "evaluation": sha256(args.evaluation),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
