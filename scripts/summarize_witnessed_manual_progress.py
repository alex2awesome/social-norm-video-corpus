#!/usr/bin/env python3
"""Validate and summarize an in-progress witnessed atomic audit ledger.

This intentionally does not compute model metrics.  An incomplete ledger may
be inspected for progress, but only the strict evaluator may produce final
audit estimates after every selected candidate has a complete judgment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_witnessed_video_asr_corpus_audit import (
        composed_positive,
        read_jsonl,
        validate_manual,
    )
else:
    from evaluate_witnessed_video_asr_corpus_audit import (
        composed_positive,
        read_jsonl,
        validate_manual,
    )


JUDGMENT_FIELDS = (
    "reaction_grounded",
    "action_before_or_overlaps_response",
    "response_targets_action",
    "responder_role",
    "response_content",
    "trigger_kind",
    "staging",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def selected_lineage(
    selection_rows: list[dict[str, Any]],
) -> dict[str, tuple[str, str]]:
    selected: dict[str, tuple[str, str]] = {}
    for clip in selection_rows:
        item_id, uid = str(clip["item_id"]), str(clip["uid"])
        for candidate in clip.get("candidates") or []:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in selected:
                raise ValueError("selection contains duplicate candidate_id")
            selected[candidate_id] = (item_id, uid)
    return selected


def summarize(
    selection_rows: list[dict[str, Any]], manual_rows: list[dict[str, str]]
) -> dict[str, Any]:
    selected = selected_lineage(selection_rows)
    manual = {row.get("candidate_id", ""): row for row in manual_rows}
    if "" in manual or len(manual) != len(manual_rows):
        raise ValueError("manual ledger has empty or duplicate candidate_id")
    if set(manual) != set(selected):
        raise ValueError("manual ledger does not exactly cover sealed selection")

    completed: list[dict[str, str]] = []
    blank = 0
    for candidate_id, row in manual.items():
        item_id, uid = selected[candidate_id]
        if row.get("item_id") != item_id or row.get("uid") != uid:
            raise ValueError(f"{candidate_id}: manual item_id/uid lineage mismatch")
        present = [bool((row.get(field) or "").strip()) for field in JUDGMENT_FIELDS]
        if any(present) and not all(present):
            raise ValueError(f"{candidate_id}: partially completed judgment")
        if all(present):
            completed.append(row)
        else:
            blank += 1

    if completed:
        validate_manual(completed)
    strict_reaction = sum(
        composed_positive(row, require_social=False) for row in completed
    )
    strict_social = sum(
        composed_positive(row, require_social=True) for row in completed
    )
    media_failure_abstentions = sum(
        row["reaction_grounded"] == "uncertain"
        and row["action_before_or_overlaps_response"] == "uncertain"
        and row["response_targets_action"] == "uncertain"
        and row["responder_role"] == "offscreen_or_unresolved"
        and row["response_content"] == "uncertain"
        and row["trigger_kind"] == "uncertain"
        and row["staging"] == "uncertain"
        for row in completed
    )
    total = len(selected)
    return {
        "kind": "witnessed_video_asr_manual_progress_v1",
        "selected_candidates": total,
        "completed_candidates": len(completed),
        "blank_candidates": blank,
        "completion_fraction": len(completed) / total if total else None,
        "manual_review_complete": len(completed) == total,
        "completed_strict_reaction_candidates_descriptive_only": strict_reaction,
        "completed_strict_social_candidates_descriptive_only": strict_social,
        "media_failure_abstentions": media_failure_abstentions,
        "final_metrics_authorized": len(completed) == total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = summarize(read_jsonl(args.selection), read_tsv(args.manual))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
