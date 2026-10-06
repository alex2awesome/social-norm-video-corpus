#!/usr/bin/env python3
"""Materialize the audited blind-episode rule as append-only shadow scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RULE_ID = "instructional_qwen_blind_episode_relaxed_v1"


def episode_without_completeness(result: dict[str, Any]) -> bool:
    """Frozen rule duplicated locally so the scale scorer is deployable alone."""
    return (
        result["visual_context"] in {"connected_episode", "roleplay_episode"}
        and result["participant_grounding"] in {
            "affected_party_present", "shared_audience_present"
        }
        and (
            result["physical_social_action"] == "yes"
            or result["quote_function"] == "situated_to_present_party"
        )
    )


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    error = row.get("error")
    eligible = not error and isinstance(result, dict)
    return {
        "item_id": row["item_id"],
        "uid": row.get("uid"),
        "pillar": "instructional",
        "rule_id": RULE_ID,
        "candidate_for_review": (
            episode_without_completeness(result) if eligible else False
        ),
        "strict_episode_pass": bool(result and result.get("episode_pass")),
        "model": row.get("model"),
        "rubric": row.get("rubric"),
        "atomic_result": result,
        "source_error": error,
        "allowed_uses": [],
        "transfer_status": "failed_transfer",
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    scored = [score_row(row) for row in rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored)
    )
    summary = {
        "kind": "instructional_blind_episode_shadow_scores_v1",
        "rule_id": RULE_ID,
        "items": len(scored),
        "errors": sum(bool(row["source_error"]) for row in scored),
        "candidates": sum(row["candidate_for_review"] for row in scored),
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
