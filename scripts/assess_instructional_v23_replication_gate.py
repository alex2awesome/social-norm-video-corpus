#!/usr/bin/env python3
"""Apply the frozen V23 maintenance gate after complete model-output review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "true", "y", "yes"}
REVIEW_FIELDS = (
    "qwen_visual_reviewed", "qwen_semantic_reviewed",
    "gemma_visual_reviewed", "gemma_semantic_reviewed",
)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def meets(metric: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        metric.get("precision") is not None
        and metric["precision"] >= float(gate["minimum_precision"])
        and metric.get("recall") is not None
        and metric["recall"] >= float(gate["minimum_recall"])
        and metric["selected"] >= int(gate["minimum_selected"])
    )


def assess(
    summary: dict[str, Any],
    preregistration: dict[str, Any],
    reviews: list[dict[str, str]],
) -> dict[str, Any]:
    gate = preregistration["maintenance_gate"]
    expected = int(gate["required_items"])
    indexes = {int(row["audit_index"]) for row in reviews}
    review_complete = (
        len(reviews) == expected
        and indexes == set(range(expected))
        and all(all(truthy(row.get(field)) for field in REVIEW_FIELDS) for row in reviews)
    )
    coverage = (
        summary.get("items") == expected
        and summary.get("manual_coverage_complete") is True
        and not any(
            items
            for stage in summary.get("model_failed_items", {}).values()
            for items in stage.values()
        )
    )
    metric = summary["pipeline_consensus"]
    retain = review_complete and coverage and meets(
        metric, gate["retain_candidate_generation_and_review_ranking"]
    )
    downgrade = (
        not retain
        and review_complete
        and coverage
        and meets(metric, gate["downgrade_to_review_ranking_only"])
    )
    action = (
        "retain_candidate_generation_and_review_ranking" if retain
        else "downgrade_to_review_ranking_only" if downgrade
        else "disable_rule"
    )
    return {
        "kind": "instructional_v23_qwen_gemma_replication_gate",
        "items": summary["items"],
        "manual_visual_demos": summary["manual_visual_demos"],
        "manual_usable_demos": summary["manual_usable_demos"],
        "pipeline_consensus": metric,
        "model_output_reviews_required": expected * len(REVIEW_FIELDS),
        "model_output_reviews_complete": review_complete,
        "model_output_coverage_complete": coverage,
        "maintenance_action": action,
        "automatic_acceptance": False,
        "acceptance_or_rejection_authority": False,
        "corpus_mutation_authorized": False,
        "all_media_retained": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--manual-output-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.manual_output_review.open() as handle:
        reviews = list(csv.DictReader(handle, delimiter="\t"))
    result = assess(
        json.loads(args.summary.read_text()),
        json.loads(args.preregistration.read_text()),
        reviews,
    )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
