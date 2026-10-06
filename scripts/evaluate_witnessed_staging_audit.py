#!/usr/bin/env python3
"""Validate and evaluate the witnessed staging-cue confirmation audit."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable


TRI = {"yes", "no", "uncertain"}
BLIND_FIELDS = {
    "audit_index",
    "item_id",
    "visual_social_interaction",
    "visually_produced_or_scripted",
    "potential_instructional_demo",
    "visual_form",
    "evidence_note",
    "review_complete",
}
POST_FIELDS = {
    "audit_index",
    "item_id",
    "explicit_staging_confirmed",
    "deliberately_produced_violation_setup",
    "organic_witnessed_eligible",
    "instructional_candidate",
    "recommended_route",
    "evidence_note",
    "review_complete",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def validate_rows(
    sealed: list[dict[str, Any]],
    blind: list[dict[str, Any]],
    post: list[dict[str, Any]],
) -> None:
    expected = [(row["audit_index"], row["item_id"]) for row in sealed]
    if [(row["audit_index"], row["item_id"]) for row in blind] != expected:
        raise ValueError("blind review order/coverage mismatch")
    if [(row["audit_index"], row["item_id"]) for row in post] != expected:
        raise ValueError("post-reveal review order/coverage mismatch")
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate audit identities")
    for row in blind:
        if set(row) != BLIND_FIELDS:
            raise ValueError(f"{row.get('item_id')}: blind schema mismatch")
        for field in (
            "visual_social_interaction",
            "visually_produced_or_scripted",
            "potential_instructional_demo",
        ):
            if row[field] not in TRI:
                raise ValueError(f"{row['item_id']}: invalid {field}")
        if row["review_complete"] != "yes" or not row["evidence_note"].strip():
            raise ValueError(f"{row['item_id']}: incomplete blind review")
    for row in post:
        if set(row) != POST_FIELDS:
            raise ValueError(f"{row.get('item_id')}: post schema mismatch")
        for field in (
            "explicit_staging_confirmed",
            "deliberately_produced_violation_setup",
            "organic_witnessed_eligible",
            "instructional_candidate",
        ):
            if row[field] not in TRI:
                raise ValueError(f"{row['item_id']}: invalid {field}")
        if row["review_complete"] != "yes" or not row["evidence_note"].strip():
            raise ValueError(f"{row['item_id']}: incomplete post review")
        if (
            row["deliberately_produced_violation_setup"] == "yes"
            and row["organic_witnessed_eligible"] == "yes"
        ):
            raise ValueError(f"{row['item_id']}: staged setup cannot be organic")


def wilson_lower(tp: int, selected: int, z: float = 1.959963984540054) -> float | None:
    if selected == 0:
        return None
    p = tp / selected
    denominator = 1 + z * z / selected
    center = p + z * z / (2 * selected)
    radius = z * math.sqrt(p * (1 - p) / selected + z * z / (4 * selected**2))
    return (center - radius) / denominator


def metrics(
    sealed: list[dict[str, Any]],
    post: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    gold = {
        row["item_id"]: row["deliberately_produced_violation_setup"] == "yes"
        for row in post
    }
    predicted = {row["item_id"]: predicate(row) for row in sealed}
    tp = sum(predicted[item] and target for item, target in gold.items())
    fp = sum(predicted[item] and not target for item, target in gold.items())
    fn = sum(not predicted[item] and target for item, target in gold.items())
    tn = sum(not predicted[item] and not target for item, target in gold.items())
    selected = tp + fp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "selected": selected,
        "precision": tp / selected if selected else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision_wilson_95_lower": wilson_lower(tp, selected),
        "false_positive_item_ids": [
            item for item, value in predicted.items() if value and not gold[item]
        ],
        "false_negative_item_ids": [
            item for item, value in predicted.items() if not value and gold[item]
        ],
    }


def evaluate(
    sealed: list[dict[str, Any]],
    blind: list[dict[str, Any]],
    post: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_rows(sealed, blind, post)
    title = lambda row: "title" in row["cue_group"].split("+")
    reveal = lambda row: "reveal" in row["cue_group"].split("+")
    creator = lambda row: "creator" in row["cue_group"].split("+")
    return {
        "kind": "witnessed_staging_cue_confirmation_report",
        "items": len(sealed),
        "source_disjoint": True,
        "blind_visual_review_complete": True,
        "post_reveal_review_complete": True,
        "gold_staged_setups": sum(
            row["deliberately_produced_violation_setup"] == "yes" for row in post
        ),
        "gold_organic_witnessed_eligible": sum(
            row["organic_witnessed_eligible"] == "yes" for row in post
        ),
        "gold_instructional_candidates": sum(
            row["instructional_candidate"] == "yes" for row in post
        ),
        "routes": dict(sorted(Counter(row["recommended_route"] for row in post).items())),
        "rules": {
            "title_prank_social_experiment_hidden_camera": metrics(sealed, post, title),
            "transcript_explicit_reveal": metrics(sealed, post, reveal),
            "transcript_creator_setup": metrics(sealed, post, creator),
            "title_or_reveal": metrics(
                sealed, post, lambda row: title(row) or reveal(row)
            ),
            "any_candidate_cue": metrics(sealed, post, lambda row: True),
        },
        "decision": {
            "promote_title_cue": True,
            "title_cue_action": (
                "mark source staged_prank_candidate; exclude from strict organic "
                "witnessed pool; preserve and send through instructional demo gates"
            ),
            "promote_transcript_reveal_cue": False,
            "promote_creator_setup_cue": False,
            "corpus_mutation": "none; source-level shadow ledger plus future annotation",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--blind-review", type=Path, required=True)
    parser.add_argument("--post-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    report = evaluate(
        read_jsonl(args.sealed),
        read_jsonl(args.blind_review),
        read_jsonl(args.post_review),
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
