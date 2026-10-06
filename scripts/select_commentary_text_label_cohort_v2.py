#!/usr/bin/env python3
"""Select fresh retained commentary sources for text-label then visual audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from scripts.sample_commentary_visual_audit import load_candidates, load_exclusions
else:
    from sample_commentary_visual_audit import load_candidates, load_exclusions


MANUAL_FIELDS = (
    "uid", "item_id", "decision", "social_actor_grounded", "concrete_behavior",
    "target_or_shared_context_grounded", "normative_stance_grounded",
    "normalized_behavior", "normalized_norm", "behavior_evidence_quote",
    "stance_evidence_quote", "manual_rationale",
)


def stable(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("query_source") or "unknown"),
        str(row.get("category") or "unknown"),
        str(row.get("signal") or "unknown"),
    )


def diverse_order(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    for key, values in groups.items():
        values.sort(key=lambda row: stable(f"{seed}:{key}", row["item_id"]))
    keys = sorted(groups, key=lambda key: stable(f"{seed}:groups", repr(key)))
    output = []
    offset = 0
    while True:
        added = False
        for key in keys:
            if offset < len(groups[key]):
                output.append(groups[key][offset])
                added = True
        if not added:
            return output
        offset += 1


def select(rows: list[dict[str, Any]], count: int, seed: str) -> list[dict[str, Any]]:
    by_platform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_platform[str(row.get("source_platform") or "unknown")].append(row)
    queues = {platform: diverse_order(values, f"{seed}:{platform}")
              for platform, values in by_platform.items()}
    platforms = sorted(queues, key=lambda value: stable(f"{seed}:platform", value))
    offsets = {platform: 0 for platform in platforms}
    output = []
    while len(output) < count:
        added = False
        for platform in platforms:
            if offsets[platform] < len(queues[platform]):
                output.append(queues[platform][offsets[platform]])
                offsets[platform] += 1
                added = True
            if len(output) == count:
                break
        if not added:
            raise ValueError("insufficient fresh retained commentary sources")
    if len({row["uid"] for row in output}) != len(output):
        raise AssertionError("commentary selection is not source-disjoint")
    return output


def transcript_context(path: Path, start: float, end: float, margin: float = 30.0) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    low, high = max(0.0, start - margin), end + margin
    return [{
        "start": float(segment["start"]), "end": float(segment["end"]),
        "text": str(segment.get("text") or ""),
    } for segment in payload.get("segments") or []
        if float(segment.get("end") or 0) >= low
        and float(segment.get("start") or 0) <= high]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discussion-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--transcript-dir", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", default="commentary-hierarchical-full-source-v2-text")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    candidates = load_candidates(
        args.discussion_dir, args.video_dir,
        load_exclusions(args.exclude_uids), args.seed,
    )
    selected = select(candidates, args.count, args.seed)
    args.out_dir.mkdir(parents=True)
    enriched = []
    for row in selected:
        value = dict(row)
        value["transcript_context"] = transcript_context(
            args.transcript_dir / f"{row['uid']}.json",
            float(row["detector_start_sec"]), float(row["detector_end_sec"]),
        )
        value["text_label_manual_reviewed"] = False
        value["script_certifies_visual_event"] = False
        enriched.append(value)
    (args.out_dir / "sealed_text_selection.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in enriched
    ))
    with (args.out_dir / "manual_text_review.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in enriched:
            writer.writerow({"uid": row["uid"], "item_id": row["item_id"]})
    summary = {
        "kind": "commentary_text_label_cohort_v2",
        "selected_sources": len(enriched),
        "source_disjoint": len({row["uid"] for row in enriched}) == len(enriched),
        "platforms": {platform: sum(row["source_platform"] == platform for row in enriched)
                      for platform in sorted({row["source_platform"] for row in enriched})},
        "manual_text_review_required": True,
        "script_certifies_visual_event": False,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }
    (args.out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
