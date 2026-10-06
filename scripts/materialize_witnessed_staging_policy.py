#!/usr/bin/env python3
"""Materialize the independently audited witnessed staging-title policy.

Rows without the title cue remain undecided.  A false value is deliberately
not an acceptance: it only means this one exclusion/reroute rule did not fire.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RULE_ID = "audited_staging_or_creator_initiated_title_v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def materialize(row: dict[str, Any]) -> dict[str, Any]:
    staging_cue = row.get("title_staging_cue") is True
    creator_initiated_cue = row.get("title_creator_initiated_candidate_cue") is True
    cue = staging_cue or creator_initiated_cue
    return {
        "uid": row["uid"],
        "rule_id": RULE_ID,
        "source_staging_title_cue": staging_cue,
        "source_creator_initiated_title_cue": creator_initiated_cue,
        "production_style": "creator_produced_candidate" if cue else None,
        # False is a proven exclusion from the strict organic witnessed pool.
        # Null means undecided by this rule, never accepted.
        "strict_organic_witnessed_eligible": False if cue else None,
        "next_review": "instructional_demo_gates" if cue else None,
        "title": row.get("title"),
        "title_matches": (row.get("matches") or {}).get("title") or [],
        "creator_initiated_title_matches": (
            (row.get("matches") or {}).get("title_creator_initiated_candidate") or []
        ),
        "source_score_error": row.get("error"),
        "policy": "preserve_original_non_destructive_reroute_ledger",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    rows = [materialize(row) for row in read_jsonl(args.scores)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    )
    summary = {
        "items": len(rows),
        "reroute_candidates": sum(
            row["strict_organic_witnessed_eligible"] is False for row in rows
        ),
        "undecided": sum(
            row["strict_organic_witnessed_eligible"] is None for row in rows
        ),
        "rule_id": RULE_ID,
        "out": str(args.out),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
