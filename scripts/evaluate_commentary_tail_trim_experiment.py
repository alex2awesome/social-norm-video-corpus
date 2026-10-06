#!/usr/bin/env python3
"""Evaluate the preregistered fixed-tail commentary remediation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_commentary_clip_plan import evaluate, read_jsonl, read_tsv, sha256
else:
    from evaluate_commentary_clip_plan import evaluate, read_jsonl, read_tsv, sha256


def evaluate_fixed_tail(
    manifest: list[dict[str, Any]],
    blind: list[dict[str, str]],
    post: list[dict[str, str]],
    preregistration: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every frozen output; uncertain and missing judgments fail closed."""
    report, accepted = evaluate(manifest, blind, post)
    parent_ids = [str(row.get("parent_candidate_id") or "") for row in manifest]
    if any(not value for value in parent_ids) or len(set(parent_ids)) != len(parent_ids):
        raise ValueError("fixed-tail manifest must contain one unique non-empty parent per item")
    expected = int(preregistration["cohort"]["parents"])
    if len(manifest) != expected:
        raise ValueError("manifest does not exactly cover preregistered cohort")
    variants = {str(row.get("variant") or "") for row in manifest}
    transform = preregistration["transform"]
    if (
        transform.get("name") != "last_half_v1"
        or "final 50 percent" not in str(transform.get("rule") or "")
        or variants != {"last_half_v1"}
    ):
        raise ValueError("manifest does not implement the preregistered fixed last-half transform")
    threshold = float(
        preregistration["manual_gate"][
            "minimum_parent_salvage_rate_for_independent_replication"
        ]
    )
    usable = len(accepted)
    rate = usable / expected
    report.update({
        "kind": "commentary_fixed_tail_trim_manual_evaluation_v1",
        "development_known_failure_cohort": True,
        "items": expected,
        "parents": expected,
        "manually_usable_fixed_tail_outputs": usable,
        "parent_salvage_rate": rate,
        "promotion_threshold": threshold,
        "replication_eligible": rate >= threshold,
        "shadow_transform_candidate_promoted": False,
        "automatic_acceptance_rule_promoted": False,
        "source_disjoint": False,
        "corpus_mutated": False,
        "corpus_disposition": None,
        "policy": "development_failure_analysis_only_preserve_all_parents",
    })
    return report, accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--post-reveal", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.accepted.exists():
        raise SystemExit("refusing to overwrite evaluation artifacts")
    report, accepted = evaluate_fixed_tail(
        read_jsonl(args.manifest),
        read_tsv(args.blind),
        read_tsv(args.post_reveal),
        json.loads(args.preregistration.read_text()),
    )
    args.accepted.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted)
    )
    report["artifact_sha256"] = {
        "manifest": sha256(args.manifest),
        "blind": sha256(args.blind),
        "post_reveal": sha256(args.post_reveal),
        "preregistration": sha256(args.preregistration),
        "accepted": sha256(args.accepted),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
