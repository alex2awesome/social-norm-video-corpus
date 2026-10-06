#!/usr/bin/env python3
"""Write read-only scene-title review scores for every instructional source."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_instructional_scene_title_cues import title_cues
except ModuleNotFoundError:
    from evaluate_instructional_scene_title_cues import title_cues  # type: ignore[no-redef]


def score_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text())
    title = str(metadata.get("title") or "")
    uid = path.parent.name
    cues = title_cues(title)
    demos = metadata.get("demos")
    if not isinstance(demos, list):
        demos = []
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    return {
        "uid": uid,
        "title": title,
        "category": str(metadata.get("category") or provenance.get("category") or ""),
        "found_by_query": str(provenance.get("found_by_query") or ""),
        "query_source": str(provenance.get("query_source") or ""),
        "demo_count": len(demos),
        **cues,
        "allowed_use": "candidate_generation_and_manual_review_priority_only",
        "automatic_acceptance": False,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, output: Path, summary_path: Path) -> dict[str, Any]:
    paths = sorted(root.glob("*/metadata.json"))
    rows = [score_metadata(path) for path in paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    selected = [row for row in rows if row["scene_title_candidate"]]
    summary = {
        "kind": "instructional_scene_title_cue_full_corpus_shadow",
        "policy": "read_only_review_priority_no_keep_reject_or_reroute",
        "sources_scored": len(rows),
        "sources_selected_for_priority_review": len(selected),
        "demo_clips_in_selected_sources": sum(row["demo_count"] for row in selected),
        "positive_cue_counts_selected": dict(
            Counter(cue for row in selected for cue in row["positive_cues"])
        ),
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
    print(json.dumps(run(args.root, args.output, args.summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
