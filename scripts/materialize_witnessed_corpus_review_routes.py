#!/usr/bin/env python3
"""Merge audited witnessed cues into one non-destructive clip routing ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


RULE_ID = "witnessed_non_destructive_review_routes_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route(
    authority_row: dict[str, Any],
    staging_by_uid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    uid = str(authority_row["uid"])
    staging = staging_by_uid.get(uid) or {}
    staged_source = (
        staging.get("strict_organic_witnessed_eligible") is False
    )
    authority_reaction = authority_row.get("authority_reaction_cue") is True

    review_routes: list[str] = []
    exclusion_reasons: list[str] = []
    if staged_source:
        review_routes.append("instructional_demo_review")
        exclusion_reasons.append("audited_staging_or_creator_title_cue_v2")
    if authority_reaction:
        review_routes.append("visible_scene_review")
        exclusion_reasons.append("authority_reaction_exact_span_cue_v3")
    if not review_routes:
        review_routes.append("strict_witnessed_unresolved_review")

    return {
        "item_id": authority_row["item_id"],
        "uid": uid,
        "clip_idx": authority_row.get("clip_idx"),
        "rule_id": RULE_ID,
        # False means excluded by an audited cue. Null means this ledger has
        # not decided the strict label. Absence of a cue is never acceptance.
        "strict_organic_witnessed_eligible": (
            False if exclusion_reasons else None
        ),
        "strict_exclusion_reasons": exclusion_reasons,
        "review_routes": review_routes,
        "source_staging_title_cue": bool(
            staging.get("source_staging_title_cue")
        ),
        "source_creator_initiated_title_cue": bool(
            staging.get("source_creator_initiated_title_cue")
        ),
        "authority_reaction_text_cue": authority_reaction,
        "authority_cue_ids": authority_row.get("authority_cue_ids") or [],
        "corpus_disposition": None,
        "policy": "append_only_shadow_preserve_all_media_and_metadata",
    }


def materialize(
    authority_rows: list[dict[str, Any]],
    staging_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    staging_by_uid = {str(row["uid"]): row for row in staging_rows}
    if len(staging_by_uid) != len(staging_rows):
        raise ValueError("staging policy requires one row per UID")
    item_ids = [str(row["item_id"]) for row in authority_rows]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("authority scores require unique item IDs")

    rows = [route(row, staging_by_uid) for row in authority_rows]
    route_counts = Counter(
        route_name for row in rows for route_name in row["review_routes"]
    )
    summary = {
        "kind": "witnessed_non_destructive_corpus_review_routes",
        "rule_id": RULE_ID,
        "clip_records": len(rows),
        "source_policy_records": len(staging_rows),
        "strict_excluded_by_audited_cue": sum(
            row["strict_organic_witnessed_eligible"] is False for row in rows
        ),
        "strict_unresolved": sum(
            row["strict_organic_witnessed_eligible"] is None for row in rows
        ),
        "route_counts": dict(sorted(route_counts.items())),
        "overlapping_reroute_cues": sum(
            len(row["review_routes"]) > 1 for row in rows
        ),
        "policy": "append_only_shadow_no_keep_reject_or_corpus_mutation",
        "warning": "strict_unresolved means undecided, never accepted",
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-scores", type=Path, required=True)
    parser.add_argument("--staging-policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.summary.exists():
        raise SystemExit("refusing to overwrite frozen output")
    rows, summary = materialize(
        read_jsonl(args.authority_scores),
        read_jsonl(args.staging_policy),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        )
    )
    summary["artifact_sha256"] = {
        "authority_scores": sha256(args.authority_scores),
        "staging_policy": sha256(args.staging_policy),
        "routes": sha256(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
