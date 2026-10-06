#!/usr/bin/env python3
"""Score all existing instructional demos with audited retro-scan priority."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


RULE_ID = "instructional_retro_query_source_priority_v1"
TRIGGER = "retro_instr_scan"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_metadata(path: Path) -> list[dict[str, Any]]:
    metadata = json.loads(path.read_text())
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    query_source = str(
        provenance.get("query_source") or metadata.get("query_source") or ""
    )
    demos = metadata.get("demos")
    if not isinstance(demos, list):
        demos = []
    uid = path.parent.name
    priority = query_source == TRIGGER
    rows = []
    for index, demo in enumerate(demos):
        if not isinstance(demo, dict):
            demo = {}
        rows.append({
            "item_id": f"instructional:{uid}:{index}",
            "uid": uid,
            "demo_index": index,
            "clip": str(demo.get("clip") or ""),
            "query_source": query_source,
            "signal_value": TRIGGER if priority else query_source,
            "review_priority": "high_review" if priority else "standard",
            "triggered": priority,
            "rule_id": RULE_ID,
            "allowed_use": "manual_review_priority_only",
            "automatic_acceptance": False,
            "corpus_disposition": None,
            "delete_media": False,
        })
    return rows


def run(root: Path, output: Path, summary_path: Path) -> dict[str, Any]:
    metadata_paths = sorted(root.glob("*/metadata.json"))
    rows = [row for path in metadata_paths for row in score_metadata(path)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = {
        "kind": "instructional_retro_query_source_full_corpus_shadow_v1",
        "policy": "append_only_review_priority_no_keep_reject_or_delete",
        "sources_scored": len(metadata_paths),
        "demo_clips_scored": len(rows),
        "priority_sources": len({row["uid"] for row in rows if row["triggered"]}),
        "priority_demo_clips": sum(row["triggered"] for row in rows),
        "query_source_counts": dict(sorted(Counter(row["query_source"] for row in rows).items())),
        "rule_id": RULE_ID,
        "automatic_acceptance": False,
        "corpus_mutated": False,
        "output_sha256": sha256(output),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise SystemExit("refusing to overwrite existing shadow-score artifacts")
    print(json.dumps(run(args.root, args.output, args.summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
