#!/usr/bin/env python3
"""Score the replicated instructional review tiers over every existing demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


NON_EXPLANATION = {"violation", "correct", "contrast"}
RETRO = "retro_instr_scan"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_metadata(path: Path) -> list[dict[str, Any]]:
    metadata = json.loads(path.read_text())
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    query_source = str(provenance.get("query_source") or metadata.get("query_source") or "")
    demos = metadata.get("demos")
    if not isinstance(demos, list):
        raise ValueError("demos is not a list")
    uid = path.parent.name
    rows = []
    for index, raw_demo in enumerate(demos):
        demo = raw_demo if isinstance(raw_demo, dict) else {}
        polarity = str(demo.get("polarity") or "unknown").strip().casefold()
        retro = query_source == RETRO
        non_explanation = polarity in NON_EXPLANATION
        if retro and non_explanation:
            tier = "tier_1_both"
        elif retro:
            # This combination was absent from both manual cohorts. Preserve
            # the independently audited retro priority and flag extrapolation.
            tier = "tier_1_retro_out_of_combination_support"
        elif non_explanation:
            tier = "tier_2_polarity_only"
        else:
            tier = "tier_3_lower_priority_preserved"
        rows.append({
            "item_id": f"instructional:{uid}:{index}",
            "uid": uid,
            "demo_index": index,
            "clip": demo.get("clip"),
            "query_source": query_source,
            "polarity": polarity,
            "instructional_retro_query_source_priority_v1": RETRO if retro else None,
            "instructional_non_explanation_review_priority_v1": polarity if non_explanation else None,
            "combined_review_tier": tier,
            "manual_review_required": True,
            "automatic_acceptance": False,
            "automatic_rejection": False,
            "corpus_disposition": None,
            "delete_media": False,
            "error": None,
        })
    return rows


def score_corpus(root: Path) -> list[dict[str, Any]]:
    rows = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        try:
            rows.extend(score_metadata(metadata_path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            rows.append({
                "item_id": f"instructional:{metadata_path.parent.name}:unscored",
                "uid": metadata_path.parent.name,
                "demo_index": None,
                "clip": None,
                "query_source": None,
                "polarity": None,
                "instructional_retro_query_source_priority_v1": None,
                "instructional_non_explanation_review_priority_v1": None,
                "combined_review_tier": "unscored_preserved",
                "manual_review_required": True,
                "automatic_acceptance": False,
                "automatic_rejection": False,
                "corpus_disposition": None,
                "delete_media": False,
                "error": f"{type(error).__name__}: {error}",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.summary.exists():
        raise SystemExit("refusing to overwrite an existing shadow artifact")
    rows = score_corpus(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    counts = Counter(row["combined_review_tier"] for row in rows)
    summary = {
        "kind": "instructional_combined_review_tiers_full_corpus_v1",
        "rows": len(rows),
        "successful_demo_rows": sum(row["error"] is None for row in rows),
        "failed_source_rows_preserved": sum(row["error"] is not None for row in rows),
        "tier_counts": dict(sorted(counts.items())),
        "output_sha256": sha256(args.out),
        "automatic_acceptance": False,
        "automatic_rejection": False,
        "corpus_mutated": False,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
