#!/usr/bin/env python3
"""Evaluate label-blind, atomic commentary microclip evidence.

The observers see only video.  A separate alignment result may compare the
literal observer intersection with the proposed label.  This evaluator is
shadow-only and never mutates corpus metadata or media.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

if __package__:
    from scripts.assign_instructional_evidence_bands import blind_event_pass
    from scripts.evaluate_commentary_scene_benchmark import (
        evaluate_predictions,
        load_jsonl,
    )
else:
    from assign_instructional_evidence_bands import blind_event_pass
    from evaluate_commentary_scene_benchmark import (
        evaluate_predictions,
        load_jsonl,
    )


def successful_by_item(rows: list[dict]) -> dict[str, dict]:
    successful = [
        row
        for row in rows
        if row.get("result") is not None and row.get("error") is None
    ]
    result = {row["item_id"]: row for row in successful}
    if len(result) != len(successful):
        raise ValueError("successful score rows contain duplicate item_id values")
    return result


def strict_text_pass(record: dict | None) -> bool:
    if not record or not isinstance(record.get("result"), dict):
        return False
    result = record["result"]
    return (
        result.get("social_norm_candidate") == "yes"
        and result.get("concrete_behavior_named") == "yes"
        and result.get("affected_other_or_shared_setting_named") == "yes"
        and result.get("norm_type")
        in {"tacit_interpersonal", "shared_public"}
    )


def atomic_window_class(
    observer_a: dict,
    observer_b: dict,
    alignment: dict,
    confidence_threshold: float,
) -> str:
    """Return exact, relabel, conflict, or low_evidence for one window."""
    a_pass = blind_event_pass(observer_a, confidence_threshold)
    b_pass = blind_event_pass(observer_b, confidence_threshold)
    result = alignment["result"]
    intersected = (
        a_pass
        and b_pass
        and result.get("both_observers_found_event") == "yes"
        and result.get("observers_describe_same_event") == "yes"
        and result.get("visual_evidence_independent_of_metadata") == "yes"
        and result.get("literal_social_behavior_visible") == "yes"
        and result.get("social_norm_domain_of_intersection")
        in {"tacit_interpersonal", "shared_public"}
    )
    if (
        intersected
        and result.get("proposed_label_kind") == "concrete_conduct"
        and result.get("proposed_label_action_visible") == "yes"
        and result.get("response_or_aftermath_only_for_proposed_label") == "no"
        and result.get("exact_label_supported") == "yes"
        and result.get("failure_reason") == "none"
    ):
        return "exact"
    if intersected and result.get("usable_after_relabel") == "yes":
        return "relabel"
    if a_pass and b_pass:
        return "semantic_conflict"
    return "low_evidence"


def evaluate_atomic_microclips(
    manual_rows: list[dict],
    window_rows: list[dict],
    observer_a_rows: list[dict],
    observer_b_rows: list[dict],
    alignment_rows: list[dict],
    *,
    text_rows: list[dict] | None = None,
    confidence_threshold: float = 0.6,
) -> dict:
    windows = {row["item_id"]: row for row in window_rows}
    if len(windows) != len(window_rows):
        raise ValueError("window manifest contains duplicate item_id values")
    observer_a = successful_by_item(observer_a_rows)
    observer_b = successful_by_item(observer_b_rows)
    alignment = successful_by_item(alignment_rows)
    text = successful_by_item(text_rows or [])

    parents: dict[str, list[dict]] = defaultdict(list)
    for window in window_rows:
        parents[window["parent_item_id"]].append(window)

    parent_counts: dict[str, dict] = {}
    for parent_item_id, parent_windows in parents.items():
        counts = {
            "windows": len(parent_windows),
            "valid_atomic_windows": 0,
            "exact_windows": 0,
            "relabel_windows": 0,
            "semantic_conflict_windows": 0,
            "low_evidence_windows": 0,
            "exact_window_ids": [],
            "relabel_window_ids": [],
        }
        for window in parent_windows:
            item_id = window["item_id"]
            if (
                item_id not in observer_a
                or item_id not in observer_b
                or item_id not in alignment
            ):
                continue
            window_class = atomic_window_class(
                observer_a[item_id],
                observer_b[item_id],
                alignment[item_id],
                confidence_threshold,
            )
            counts["valid_atomic_windows"] += 1
            counts[f"{window_class}_windows"] += 1
            if window_class in {"exact", "relabel"}:
                counts[f"{window_class}_window_ids"].append(item_id)
        counts["strict_text_pass"] = strict_text_pass(text.get(parent_item_id))
        parent_counts[parent_item_id] = counts

    manual_ids = {row["item_id"] for row in manual_rows}
    parent_ids = set(parents)
    if manual_ids != parent_ids:
        raise ValueError(
            "manual/window parent item IDs differ: "
            f"manual_only={sorted(manual_ids - parent_ids)}, "
            f"window_only={sorted(parent_ids - manual_ids)}"
        )
    rules: dict[str, dict[str, bool | None]] = {}
    for threshold in (1, 2, 3):
        for prefix, predicate in (
            (
                "atomic_exact",
                lambda counts, threshold=threshold: counts["exact_windows"]
                >= threshold,
            ),
            (
                "atomic_exact_and_text_strict",
                lambda counts, threshold=threshold: counts["exact_windows"]
                >= threshold
                and counts["strict_text_pass"],
            ),
            (
                "atomic_any_recovery",
                lambda counts, threshold=threshold: (
                    counts["exact_windows"] + counts["relabel_windows"]
                )
                >= threshold,
            ),
        ):
            rules[f"{prefix}_at_least_{threshold}"] = {
                item_id: (
                    None
                    if item_id not in parent_counts
                    or parent_counts[item_id]["valid_atomic_windows"] == 0
                    else predicate(parent_counts[item_id])
                )
                for item_id in manual_ids
            }

    return {
        "status": "shadow_only_do_not_promote_without_source_disjoint_replication",
        "confidence_threshold": confidence_threshold,
        "manual_items": len(manual_rows),
        "window_items": len(window_rows),
        "observer_a_valid_items": len(observer_a),
        "observer_b_valid_items": len(observer_b),
        "alignment_valid_items": len(alignment),
        "text_valid_items": len(text),
        "rules": {
            name: evaluate_predictions(manual_rows, predictions)
            for name, predictions in rules.items()
        },
        "parent_window_counts": parent_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--observer-a", type=Path, required=True)
    parser.add_argument("--observer-b", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--text")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_atomic_microclips(
        load_jsonl(args.manual),
        load_jsonl(args.windows),
        load_jsonl(args.observer_a),
        load_jsonl(args.observer_b),
        load_jsonl(args.alignment),
        text_rows=load_jsonl(Path(args.text)) if args.text else None,
        confidence_threshold=args.confidence_threshold,
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                name: metrics["all_target_candidates"]
                for name, metrics in report["rules"].items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
