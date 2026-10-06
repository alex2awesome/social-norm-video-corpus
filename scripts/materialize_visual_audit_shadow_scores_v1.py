#!/usr/bin/env python3
"""Materialize only the atomic evidence actually established by sparse review.

Unreviewed atoms stay ``uncertain``.  This deliberately demonstrates that a
three-frame audit may veto a clear failure family but cannot manufacture a
complete keep candidate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

try:
    from scripts.cross_pillar_shadow_supervision_v1 import FAMILY_FIELDS, score_record
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.cross_pillar_shadow_supervision_v1 import FAMILY_FIELDS, score_record


def atomic_from_review(review: dict) -> dict:
    pillar = review["pillar"]
    row = {"item_id": f"{pillar}:{review['uid']}", "pillar": pillar,
           "manual_review_provenance": "20260807_three_frame_source_disjoint"}
    for fields in FAMILY_FIELDS[pillar].values():
        for field in fields:
            row[field] = "uncertain"
    decision = review["visual_target"]
    failure = review["failure"]
    if pillar == "instructional":
        row["demo_event_observable"] = decision
        row["demo_action_complete"] = decision if decision == "no" else "uncertain"
        if failure in {"non_social_technical_procedure", "off_topic_business_lecture",
                       "off_topic_technical", "health_technical_topic",
                       "health_topic_no_social_demo", "sports_instruction_no_social_demo"}:
            row["is_social_norm"] = "no"
    elif pillar == "witnessed":
        if failure == "authority_not_bystander":
            row["reactor_non_authority"] = "no"
            row["distinct_bystander"] = "no"
        elif failure in {"staged_reaction", "scripted_nonhuman"}:
            row["organic_authenticity"] = "no"
        elif failure in {"accident_without_bystander_reaction", "virtual_event"}:
            row["is_social_norm"] = "no"
        elif decision == "no":
            row["distinct_bystander"] = "no"
    else:
        row["event_visible"] = decision
        # Sparse frames can establish visibility, not exact temporal bounds.
        row["event_temporally_localized"] = "no" if decision == "no" else "uncertain"
        if failure in {"sports_rule_not_social_norm_target", "virtual_or_explanation_only"}:
            row["is_social_norm"] = "no"
    return score_record(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    reviews = [json.loads(x) for x in args.ledger.read_text().splitlines() if x.strip()]
    scored = [atomic_from_review(x) for x in reviews]
    args.out.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in scored))
    print(json.dumps(Counter(x["review_band"] for x in scored), sort_keys=True))


if __name__ == "__main__":
    main()
