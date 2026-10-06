#!/usr/bin/env python3
"""Evaluate AND/OR ensembles over paired microclip VLM results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_visual_scene_scores import metrics  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    windows = {row["item_id"]: row for row in load_jsonl(args.window_manifest)}
    left = {row["item_id"]: row for row in load_jsonl(args.left) if row.get("result")}
    right = {row["item_id"]: row for row in load_jsonl(args.right) if row.get("result")}
    paired = sorted(left.keys() & right.keys() & windows.keys())
    fields = {
        "situated_social_scenario_visible": "gold_social_scene_visible",
        "proposed_norm_plausibly_demonstrated": "gold_label_matched_visible",
        "witnessed_violation_action_visible": "gold_label_matched_visible",
    }
    report = {"paired_windows": len(paired), "fields": {}}
    for field, target in fields.items():
        for operator in ("and", "or"):
            grouped: dict[str, dict] = defaultdict(
                lambda: {"positive_indexes": [], "gold": False, "valid": 0}
            )
            for item_id in paired:
                lval = left[item_id]["result"].get(field)
                rval = right[item_id]["result"].get(field)
                if lval not in {"yes", "no"} or rval not in {"yes", "no"}:
                    continue
                positive = (
                    lval == "yes" and rval == "yes"
                    if operator == "and"
                    else lval == "yes" or rval == "yes"
                )
                parent = windows[item_id]["parent_item_id"]
                grouped[parent]["gold"] = bool(left[item_id][target])
                grouped[parent]["valid"] += 1
                if positive:
                    grouped[parent]["positive_indexes"].append(
                        windows[item_id]["window_index"]
                    )
            parents = list(grouped.values())
            rules = {}
            for count in (1, 2, 3):
                predictions = [
                    len(parent["positive_indexes"]) >= count for parent in parents
                ]
                rules[f"at_least_{count}_positive_windows"] = metrics(
                    [parent["gold"] for parent in parents], predictions
                )
            report["fields"][f"{field}_{operator}"] = {
                "target": target,
                "parents": len(parents),
                "rules": rules,
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
