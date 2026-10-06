#!/usr/bin/env python3
"""Export every materialized instructional demo as a selection population."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def export_population(instructional_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    missing_clips = []
    item_ids: set[str] = set()
    for metadata_path in sorted(instructional_dir.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        uid = str(metadata.get("video_id") or metadata_path.parent.name)
        provenance = metadata.get("provenance") or {}
        category = metadata.get("category") or provenance.get("category")
        for index, demo in enumerate(metadata.get("demos") or []):
            clip_name = demo.get("clip")
            if not isinstance(clip_name, str) or not clip_name.strip():
                continue
            clip = metadata_path.parent / clip_name
            item_id = f"instructional:{uid}:{index}"
            if item_id in item_ids:
                raise ValueError(f"duplicate item_id: {item_id}")
            item_ids.add(item_id)
            if not clip.is_file() or clip.stat().st_size <= 0:
                missing_clips.append(item_id)
                continue
            rows.append({
                "item_id": item_id,
                "uid": uid,
                "demo_index": index,
                "pillar": "instructional",
                "source_platform": uid.split("__", 1)[0],
                "source_clip": str(clip.resolve()),
                "category": category,
                "polarity": demo.get("polarity") or "unknown",
                "norm": demo.get("norm"),
                "explanation": demo.get("explanation"),
                "start_quote": demo.get("start_quote"),
                "end_quote": demo.get("end_quote"),
                "start_sec": demo.get("start"),
                "end_sec": demo.get("end"),
                "title": metadata.get("title"),
                "policy": "audit_population_only_no_keep_reject_or_corpus_mutation",
            })
    summary = {
        "kind": "instructional_materialized_demo_population_v1",
        "sources": len({row["uid"] for row in rows}),
        "demos": len(rows),
        "missing_clip_records": len(missing_clips),
        "missing_clip_item_ids_sample": missing_clips[:20],
        "polarities": dict(sorted(Counter(str(row["polarity"]) for row in rows).items())),
        "categories": dict(sorted(Counter(str(row["category"] or "unknown") for row in rows).items())),
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instructional-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = export_population(args.instructional_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
