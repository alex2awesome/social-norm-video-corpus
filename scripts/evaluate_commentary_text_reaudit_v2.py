#!/usr/bin/env python3
"""Validate an append-only second adjudication of commentary text labels.

The first manual pass is immutable.  This stricter pass asks a different
question: does the transcript ground a particular occurred social event well
enough to seed a visual search?  Generic definitions and aggregate discussion
may remain retrieval leads, but they are not strict event-label candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DECISIONS = {"accept", "reject"}
OCCURRENCES = {
    "bounded_occurrence",
    "repeated_specific_occurrence",
    "generic_or_hypothetical",
    "ambiguous_or_unresolved",
    "no_candidate_event",
}
SPECIFICITY = {"specific", "underspecified", "none"}
STANCE = {
    "unambiguous_external",
    "contemporaneous_bystander",
    "affected_party_objection",
    "contested",
    "missing_or_unresolved",
}
ROUTES = {
    "strict_visual_search",
    "instructional_definition_retrieval",
    "exploratory_contested",
    "exploratory_ambiguous",
    "none",
}
STRICT_OCCURRENCES = {"bounded_occurrence", "repeated_specific_occurrence"}
STRICT_STANCE = {
    "unambiguous_external",
    "contemporaneous_bystander",
    "affected_party_objection",
}
REQUIRED = {
    "item_id",
    "v1_decision",
    "strict_event_label_decision",
    "occurrence_grounding",
    "behavior_specificity",
    "stance_quality",
    "visual_search_route",
    "corrected_behavior",
    "corrected_norm",
    "agrees_with_v1",
    "manual_rationale",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(
    source_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    expected_source_sha256: str,
    observed_source_sha256: str,
) -> dict[str, Any]:
    if observed_source_sha256 != expected_source_sha256:
        raise ValueError("source gold hash does not match the sealed re-audit manifest")
    source = {str(row.get("item_id") or ""): row for row in source_rows}
    review = {str(row.get("item_id") or ""): row for row in review_rows}
    if not source or "" in source or len(source) != len(source_rows):
        raise ValueError("source gold has missing or duplicate item_id")
    if "" in review or len(review) != len(review_rows):
        raise ValueError("re-audit has missing or duplicate item_id")
    if set(source) != set(review):
        raise ValueError("re-audit must exactly cover the immutable source gold")

    for item_id, row in review.items():
        missing = REQUIRED - set(row)
        if missing:
            raise ValueError(f"{item_id}: missing fields {sorted(missing)}")
        if row["v1_decision"] != source[item_id].get("decision"):
            raise ValueError(f"{item_id}: v1 decision lineage mismatch")
        if row["strict_event_label_decision"] not in DECISIONS:
            raise ValueError(f"{item_id}: invalid strict decision")
        if row["occurrence_grounding"] not in OCCURRENCES:
            raise ValueError(f"{item_id}: invalid occurrence grounding")
        if row["behavior_specificity"] not in SPECIFICITY:
            raise ValueError(f"{item_id}: invalid behavior specificity")
        if row["stance_quality"] not in STANCE:
            raise ValueError(f"{item_id}: invalid stance quality")
        if row["visual_search_route"] not in ROUTES:
            raise ValueError(f"{item_id}: invalid visual-search route")
        if not isinstance(row["agrees_with_v1"], bool):
            raise ValueError(f"{item_id}: agrees_with_v1 must be boolean")
        if not str(row["manual_rationale"]).strip():
            raise ValueError(f"{item_id}: manual rationale is required")

        v1_accept = source[item_id].get("decision") in {"accept", "accept_after_relabel"}
        v2_accept = row["strict_event_label_decision"] == "accept"
        if row["agrees_with_v1"] != (v1_accept == v2_accept):
            raise ValueError(f"{item_id}: agreement flag is inconsistent")
        if v2_accept:
            if row["occurrence_grounding"] not in STRICT_OCCURRENCES:
                raise ValueError(f"{item_id}: strict accept lacks occurred event")
            if row["behavior_specificity"] != "specific":
                raise ValueError(f"{item_id}: strict accept lacks specific behavior")
            if row["stance_quality"] not in STRICT_STANCE:
                raise ValueError(f"{item_id}: strict accept lacks resolved stance")
            if row["visual_search_route"] != "strict_visual_search":
                raise ValueError(f"{item_id}: strict accept has non-strict route")
            if not str(row["corrected_behavior"]).strip() or not str(row["corrected_norm"]).strip():
                raise ValueError(f"{item_id}: strict accept lacks corrected labels")
        elif row["visual_search_route"] == "strict_visual_search":
            raise ValueError(f"{item_id}: rejected row cannot enter strict visual search")

    decision_counts = Counter(row["strict_event_label_decision"] for row in review_rows)
    route_counts = Counter(row["visual_search_route"] for row in review_rows)
    v1_accepts = sum(
        row.get("decision") in {"accept", "accept_after_relabel"} for row in source_rows
    )
    v2_accepts = decision_counts["accept"]
    retained_v1_accepts = sum(
        row["strict_event_label_decision"] == "accept"
        and source[row["item_id"]].get("decision") in {"accept", "accept_after_relabel"}
        for row in review_rows
    )
    return {
        "kind": "commentary_text_reaudit_v2",
        "source_items": len(source_rows),
        "reaudited_items": len(review_rows),
        "coverage": len(review_rows) / len(source_rows),
        "source_sha256": observed_source_sha256,
        "v1_accepted": v1_accepts,
        "strict_event_accepted": v2_accepts,
        "strict_event_fraction": v2_accepts / len(review_rows),
        "v1_accept_retention": retained_v1_accepts / v1_accepts if v1_accepts else None,
        "decisions": dict(sorted(decision_counts.items())),
        "routes": dict(sorted(route_counts.items())),
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gold", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected_hash = str(manifest.get("source_gold_sha256") or "")
    summary = validate(
        read_jsonl(args.source_gold),
        read_jsonl(args.review),
        expected_hash,
        file_sha256(args.source_gold),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
