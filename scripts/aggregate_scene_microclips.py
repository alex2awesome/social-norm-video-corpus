#!/usr/bin/env python3
"""Aggregate window-level VLM judgments without changing source labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--vlm-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    windows = {r["item_id"]: r for r in load_jsonl(args.window_manifest)}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in load_jsonl(args.vlm_results):
        window = windows.get(result["item_id"])
        if not window:
            continue
        grouped[window["parent_item_id"]].append(
            {
                "window_item_id": result["item_id"],
                "window_index": window["window_index"],
                "window_start_sec": window["window_start_sec"],
                "window_end_sec": window["window_end_sec"],
                "result": result.get("result"),
                "error": result.get("error"),
            }
        )

    with args.out.open("w") as handle:
        for parent_item_id, rows in sorted(grouped.items()):
            valid = [row for row in rows if row["result"]]
            event_field = (
                "situated_social_scenario_visible"
                if valid
                and "situated_social_scenario_visible" in valid[0]["result"]
                else "concrete_behavior_event_visible"
            )
            label_field = (
                "proposed_norm_plausibly_demonstrated"
                if valid
                and "proposed_norm_plausibly_demonstrated" in valid[0]["result"]
                else "proposed_label_event_visible"
            )
            event_yes = [
                row
                for row in valid
                if row["result"].get(event_field) == "yes"
            ]
            label_yes = [
                row
                for row in valid
                if row["result"].get(label_field) == "yes"
            ]
            record = {
                "parent_item_id": parent_item_id,
                "windows_expected": sum(
                    1
                    for window in windows.values()
                    if window["parent_item_id"] == parent_item_id
                ),
                "windows_returned": len(rows),
                "windows_valid": len(valid),
                "event_field": event_field,
                "label_field": label_field,
                "any_concrete_behavior_event_visible": bool(event_yes),
                "any_proposed_label_event_visible": bool(label_yes),
                "event_positive_windows": event_yes,
                "label_positive_windows": label_yes,
                "windows": sorted(rows, key=lambda row: row["window_index"]),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
