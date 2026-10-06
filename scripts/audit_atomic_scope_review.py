#!/usr/bin/env python3
"""Verify complete human review of an atomic social-scope shadow cohort."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from weak_label_contract import derive_scope_labels


ATOMIC_FIELDS = (
    "actor_kind",
    "behavior_kind",
    "affected_context_kind",
    "expectation_kind",
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(source_results: Path, review_tsv: Path) -> dict:
    source = load_jsonl(source_results)
    with review_tsv.open(newline="") as handle:
        reviews = list(csv.DictReader(handle, delimiter="\t"))
    source_by_id = {row["item_id"]: row for row in source}
    review_by_id = {row["item_id"]: row for row in reviews}
    if len(source_by_id) != len(source) or len(review_by_id) != len(reviews):
        raise ValueError("duplicate item_id")
    if set(source_by_id) != set(review_by_id):
        raise ValueError("review must cover every source item exactly once")

    changes = Counter()
    decisions = Counter()
    accepted = 0
    accepted_scope_failures = []
    for item_id, review in review_by_id.items():
        derived = derive_scope_labels({key: review[key] for key in ATOMIC_FIELDS})
        if review.get("derived_is_social_norm") != derived["is_social_norm"]:
            raise ValueError(f"incorrect derived_is_social_norm for {item_id}")
        old = source_by_id[item_id].get("is_social_norm")
        changes[(str(old), derived["is_social_norm"])] += 1
        decision = str(source_by_id[item_id].get("decision"))
        decisions[decision] += 1
        if decision in {"accept", "accept_with_repairs", "accept_after_relabel", "accept_after_splice"}:
            accepted += 1
            if derived["is_social_norm"] != "yes":
                accepted_scope_failures.append(item_id)
        if not review.get("manual_scope_reason", "").strip():
            raise ValueError(f"missing manual_scope_reason for {item_id}")
    if accepted_scope_failures:
        raise ValueError(
            "atomic scope rejects previously accepted items: "
            + ", ".join(accepted_scope_failures)
        )
    return {
        "reviewed": len(reviews),
        "coverage": 1.0,
        "accepted_source_items": accepted,
        "accepted_scope_failures": 0,
        "source_decisions": dict(sorted(decisions.items())),
        "old_to_derived_scope": {
            f"{old}->{new}": count
            for (old, new), count in sorted(changes.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_results", type=Path)
    parser.add_argument("review_tsv", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.source_results, args.review_tsv), sort_keys=True))


if __name__ == "__main__":
    main()
