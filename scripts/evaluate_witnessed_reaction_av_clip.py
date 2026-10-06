#!/usr/bin/env python3
"""Evaluate candidate-window AV judgments at clip-level reaction targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_witnessed_reaction_atomic_candidates import metric
    from scripts.evaluate_witnessed_reaction_external_holdout import manual_target
    from scripts.witnessed_reaction_av_contract import (
        candidate_positive as strict_candidate_positive,
    )
else:
    from evaluate_witnessed_reaction_atomic_candidates import metric
    from evaluate_witnessed_reaction_external_holdout import manual_target
    from witnessed_reaction_av_contract import (
        candidate_positive as strict_candidate_positive,
    )


SOCIAL_EXPECTATIONS = {"interpersonal_treatment", "shared_coordination"}


def candidate_positive(
    result: dict[str, Any], *, include_authority: bool, require_social: bool
) -> bool:
    """Legacy reaction-retrieval target; authenticity is downstream here."""
    return strict_candidate_positive(
        result,
        include_authority=include_authority,
        require_social=require_social,
        require_unstaged=False,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(manual_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]]) -> dict[str, Any]:
    manual = {str(row["item_id"]): row for row in manual_rows}
    if len(manual) != len(manual_rows):
        raise ValueError("duplicate manual item_id")
    reports = {}
    for include_authority in (False, True):
        for require_social in (False, True):
            name = ("extended" if include_authority else "strict_bystander") + ("_social" if require_social else "_reaction")
            gold = {
                item: manual_target(row, include_authority=include_authority)
                and (not require_social or row.get("expectation_kind") in SOCIAL_EXPECTATIONS)
                for item, row in manual.items()
            }
            predicted = {item: False for item in manual}
            for row in model_rows:
                item = str(row["item_id"])
                if item not in predicted or row.get("error"):
                    continue
                predicted[item] = predicted[item] or candidate_positive(
                    row.get("result") or {},
                    include_authority=include_authority,
                    require_social=require_social,
                )
            reports[name] = metric(gold, predicted)
    return {
        "kind": "witnessed_reaction_av_clip_audit_v1",
        "policy": "reaction_retrieval_audit_only_authenticity_and_clip_quality_are_downstream",
        "manual_clips": len(manual),
        "clips_with_candidates": len({str(row["item_id"]) for row in model_rows}),
        "rules": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(read_jsonl(args.manual), read_jsonl(args.model))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
