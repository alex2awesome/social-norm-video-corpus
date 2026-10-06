#!/usr/bin/env python3
"""Assign append-only review bands from blind video evidence and alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def successful_by_item(rows: list[dict]) -> dict[str, dict]:
    return {
        row["item_id"]: row
        for row in rows
        if row.get("result") is not None and row.get("error") is None
    }


def blind_event_pass(record: dict, confidence_threshold: float = 0.6) -> bool:
    result = record["result"]
    return (
        result.get("visually_observable_event") == "yes"
        and result.get("metadata_needed_to_identify_action") == "no"
        and result.get("presentation_or_context_only") == "no"
        and result.get("actor_visible_description") not in {"", "none", None}
        and result.get("action_or_situated_speech_description")
        not in {"", "none", None}
        and result.get("affected_party_or_shared_setting_description")
        not in {"", "none", None}
        and float(result.get("confidence") or 0) >= confidence_threshold
    )


def assign_band(
    observer_a: dict,
    observer_b: dict,
    alignment: dict,
    confidence_threshold: float = 0.6,
) -> str:
    a_pass = blind_event_pass(observer_a, confidence_threshold)
    b_pass = blind_event_pass(observer_b, confidence_threshold)
    result = alignment["result"]
    aligned = (
        a_pass
        and b_pass
        and result.get("both_observers_found_event") == "yes"
        and result.get("observers_describe_same_event") == "yes"
        and result.get("visual_evidence_independent_of_metadata") == "yes"
        and result.get("usable_after_relabel") == "yes"
    )
    if aligned and result.get("exact_label_supported") == "yes":
        return "blind_dual_exact_candidate"
    if aligned:
        return "blind_dual_relabel_candidate"
    if a_pass and b_pass:
        return "blind_event_semantic_conflict_review"
    if a_pass or b_pass:
        return "single_blind_event_review"
    return "low_evidence_review"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observer-a", type=Path, required=True)
    parser.add_argument("--observer-b", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    args = parser.parse_args()

    observer_a = successful_by_item(load_jsonl(args.observer_a))
    observer_b = successful_by_item(load_jsonl(args.observer_b))
    alignment = successful_by_item(load_jsonl(args.alignment))
    item_ids = sorted(set(observer_a) & set(observer_b) & set(alignment))
    rows = []
    for item_id in item_ids:
        rows.append(
            {
                "item_id": item_id,
                "band": assign_band(
                    observer_a[item_id],
                    observer_b[item_id],
                    alignment[item_id],
                    args.confidence_threshold,
                ),
                "confidence_threshold": args.confidence_threshold,
                "observer_a_model": observer_a[item_id]["model"],
                "observer_b_model": observer_b[item_id]["model"],
                "alignment_model": alignment[item_id]["alignment_model"],
                "rubric_version": "instructional_blind_evidence_bands_v1",
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
