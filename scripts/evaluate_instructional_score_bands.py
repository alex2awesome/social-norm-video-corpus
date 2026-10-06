#!/usr/bin/env python3
"""Evaluate non-destructive instructional score bands on frozen manual gold."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_by_item(rows: list[dict], rubric: str) -> dict[str, dict]:
    return {
        row["item_id"]: row
        for row in rows
        if row.get("rubric_version") == rubric and row.get("error") is None
    }


def assign_band(visual: dict, transcript: dict | None) -> str:
    """Assign a review band; no band is a destructive corpus disposition."""
    vr = visual["result"]
    tr = transcript["result"] if transcript else None
    if vr["target_pillar_signal_visible"] == "yes":
        if tr is None:
            return "visual_only_candidate"
        if (
            tr["specific_hypothesis_supported"] == "yes"
            and tr["demonstration_or_example_language_present"] == "yes"
        ):
            return "audiovisual_agreement"
        return "semantic_conflict_review"
    if (
        vr["social_behavior_scene_visible"] == "yes"
        and tr is not None
        and tr["social_norm_topic_supported"] == "yes"
        and tr["demonstration_or_example_language_present"] == "yes"
        and tr["situated_dialogue_or_action_cues_present"] == "yes"
    ):
        return "localized_rescue_review"
    if vr["social_behavior_scene_visible"] == "yes":
        return "off_target_or_unlocalized_review"
    return "low_visual_evidence_review"


def summarize(records: list[dict]) -> dict:
    by_band: dict[str, Counter] = {}
    for row in records:
        counts = by_band.setdefault(row["band"], Counter())
        counts["n"] += 1
        counts["gold_pass"] += int(row["gold_target_source_pass"])
        counts["gold_fail"] += int(not row["gold_target_source_pass"])
    return {band: dict(counts) for band, counts in sorted(by_band.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--visual-rubric", default="contact_sheet_v4")
    parser.add_argument("--transcript-rubric", default="transcript_gate_v2")
    args = parser.parse_args()

    manifest = {
        row["item_id"]: row
        for row in load_jsonl(args.manifest)
        if row["pillar"] == "instructional"
    }
    visual = latest_by_item(load_jsonl(args.visual), args.visual_rubric)
    transcript = latest_by_item(load_jsonl(args.transcript), args.transcript_rubric)
    records = []
    for item_id, gold in manifest.items():
        if item_id not in visual:
            continue
        records.append(
            {
                "item_id": item_id,
                "band": assign_band(visual[item_id], transcript.get(item_id)),
                "gold_target_source_pass": gold["gold_target_source_pass"],
                "has_transcript": item_id in transcript,
            }
        )
    payload = {
        "status": "calibration_only_not_independent_validation",
        "n": len(records),
        "visual_rubric": args.visual_rubric,
        "transcript_rubric": args.transcript_rubric,
        "bands": summarize(records),
        "records": sorted(records, key=lambda row: row["item_id"]),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
