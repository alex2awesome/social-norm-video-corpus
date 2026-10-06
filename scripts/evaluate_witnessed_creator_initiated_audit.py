#!/usr/bin/env python3
"""Evaluate the refined creator-initiated title cue on a frozen manual audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_witnessed_staging_audit import (
        read_jsonl,
        validate_rows,
        wilson_lower,
    )
    from scripts.score_witnessed_staging_cues import TITLE_CREATOR_INITIATED
except ModuleNotFoundError:  # Support direct ``python scripts/<name>.py`` use.
    from evaluate_witnessed_staging_audit import read_jsonl, validate_rows, wilson_lower
    from score_witnessed_staging_cues import TITLE_CREATOR_INITIATED


def evaluate(
    sealed: list[dict[str, Any]],
    blind: list[dict[str, Any]],
    post: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_rows(sealed, blind, post)
    gold = {
        row["item_id"]: row["deliberately_produced_violation_setup"] == "yes"
        for row in post
    }
    predicted = {
        row["item_id"]: bool(TITLE_CREATOR_INITIATED.search(row.get("title", "")))
        for row in sealed
    }
    tp = sum(predicted[item] and target for item, target in gold.items())
    fp = sum(predicted[item] and not target for item, target in gold.items())
    fn = sum(not predicted[item] and target for item, target in gold.items())
    tn = sum(not predicted[item] and not target for item, target in gold.items())
    selected = tp + fp
    precision = tp / selected if selected else None
    lower = wilson_lower(tp, selected)
    audited_pass = bool(
        selected >= 30
        and precision is not None
        and precision >= 0.95
        and lower is not None
        and lower >= 0.90
    )
    return {
        "kind": "witnessed_creator_initiated_title_confirmation_report",
        "items": len(sealed),
        "source_disjoint": True,
        "blind_visual_review_complete": True,
        "post_reveal_review_complete": True,
        "gold_deliberately_produced_setups": sum(gold.values()),
        "refined_rule": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "selected": selected,
            "precision": precision,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision_wilson_95_lower": lower,
            "false_positive_item_ids": [
                item for item, value in predicted.items() if value and not gold[item]
            ],
            "excluded_false_positive_item_ids": [
                item for item, value in predicted.items() if not value and not gold[item]
            ],
        },
        "decision": {
            "audit_threshold": {
                "minimum_selected": 30,
                "minimum_precision": 0.95,
                "minimum_precision_wilson_95_lower": 0.90,
            },
            "promote_refined_cue": audited_pass,
            "action": (
                "annotate source as creator_initiated_candidate; exclude from strict "
                "organic witnessed eligibility; preserve and run instructional demo gates"
                if audited_pass
                else "ranking-only; collect more audit evidence"
            ),
            "corpus_mutation": "none; preserve all source videos and clips",
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
