#!/usr/bin/env python3
"""Evaluate parent-level rules over overlapping commentary microclips."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

if __package__:
    from scripts.evaluate_commentary_scene_benchmark import (
        evaluate_predictions,
        load_jsonl,
        predict,
    )
else:
    from evaluate_commentary_scene_benchmark import (
        evaluate_predictions,
        load_jsonl,
        predict,
    )


def consecutive_run(indexes: set[int]) -> int:
    best = current = 0
    previous = None
    for index in sorted(indexes):
        current = current + 1 if previous is not None and index == previous + 1 else 1
        best = max(best, current)
        previous = index
    return best


def evaluate_microclips(
    manual_rows: list[dict],
    window_rows: list[dict],
    primary_rows: list[dict],
    secondary_rows: list[dict],
) -> dict:
    windows = {row["item_id"]: row for row in window_rows}
    if len(windows) != len(window_rows):
        raise ValueError("window manifest contains duplicate item_id values")
    primary = {row["item_id"]: row for row in primary_rows}
    secondary = {row["item_id"]: row for row in secondary_rows}
    parents: dict[str, list[dict]] = defaultdict(list)
    for window in window_rows:
        parents[window["parent_item_id"]].append(window)

    counts: dict[str, dict] = {}
    for parent_item_id, rows in parents.items():
        first_yes: set[int] = set()
        second_yes: set[int] = set()
        dual_yes: set[int] = set()
        errors = 0
        for window in rows:
            item_id = window["item_id"]
            first = predict(primary.get(item_id, {"error": "missing"}), "v3_conditioned")
            second = predict(
                secondary.get(item_id, {"error": "missing"}), "v3_conditioned"
            )
            if first is None or second is None:
                errors += 1
                continue
            index = int(window["window_index"])
            if first:
                first_yes.add(index)
            if second:
                second_yes.add(index)
            if first and second:
                dual_yes.add(index)
        counts[parent_item_id] = {
            "windows": len(rows),
            "valid_dual_windows": len(rows) - errors,
            "errors": errors,
            "primary_positive_windows": len(first_yes),
            "secondary_positive_windows": len(second_yes),
            "dual_positive_windows": len(dual_yes),
            "primary_consecutive_run": consecutive_run(first_yes),
            "secondary_consecutive_run": consecutive_run(second_yes),
            "dual_consecutive_run": consecutive_run(dual_yes),
        }

    rule_predictions: dict[str, dict[str, bool | None]] = {}
    for threshold in (1, 2, 3):
        rule_predictions[f"primary_at_least_{threshold}"] = {
            item_id: (
                None
                if item_id not in counts or counts[item_id]["valid_dual_windows"] == 0
                else counts[item_id]["primary_positive_windows"] >= threshold
            )
            for item_id in {row["item_id"] for row in manual_rows}
        }
        rule_predictions[f"secondary_at_least_{threshold}"] = {
            item_id: (
                None
                if item_id not in counts or counts[item_id]["valid_dual_windows"] == 0
                else counts[item_id]["secondary_positive_windows"] >= threshold
            )
            for item_id in {row["item_id"] for row in manual_rows}
        }
        rule_predictions[f"dual_same_window_at_least_{threshold}"] = {
            item_id: (
                None
                if item_id not in counts or counts[item_id]["valid_dual_windows"] == 0
                else counts[item_id]["dual_positive_windows"] >= threshold
            )
            for item_id in {row["item_id"] for row in manual_rows}
        }
        rule_predictions[f"dual_parent_each_at_least_{threshold}"] = {
            item_id: (
                None
                if item_id not in counts or counts[item_id]["valid_dual_windows"] == 0
                else (
                    counts[item_id]["primary_positive_windows"] >= threshold
                    and counts[item_id]["secondary_positive_windows"] >= threshold
                )
            )
            for item_id in {row["item_id"] for row in manual_rows}
        }
        rule_predictions[f"dual_same_window_consecutive_{threshold}"] = {
            item_id: (
                None
                if item_id not in counts or counts[item_id]["valid_dual_windows"] == 0
                else counts[item_id]["dual_consecutive_run"] >= threshold
            )
            for item_id in {row["item_id"] for row in manual_rows}
        }

    return {
        "status": "shadow_only_do_not_promote_without_source_disjoint_replication",
        "manual_items": len(manual_rows),
        "window_items": len(window_rows),
        "primary_items": len(primary_rows),
        "secondary_items": len(secondary_rows),
        "rules": {
            name: evaluate_predictions(manual_rows, predictions)
            for name, predictions in rule_predictions.items()
        },
        "parent_window_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_microclips(
        load_jsonl(args.manual),
        load_jsonl(args.windows),
        load_jsonl(args.primary),
        load_jsonl(args.secondary),
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        name: metrics["all_target_candidates"]
        for name, metrics in report["rules"].items()
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
