#!/usr/bin/env python3
"""Evaluate parent-level aggregation of overlapping VLM microclip judgments."""

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


def longest_run(indexes: list[int]) -> int:
    if not indexes:
        return 0
    best = current = 1
    for previous, value in zip(indexes, indexes[1:]):
        current = current + 1 if value == previous + 1 else 1
        best = max(best, current)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--vlm-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    windows = {row["item_id"]: row for row in load_jsonl(args.window_manifest)}
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for result in load_jsonl(args.vlm_results):
        window = windows.get(result["item_id"])
        if window and result.get("result"):
            grouped[window["parent_item_id"]].append((window, result))

    fields = {
        "situated_social_scenario_visible": (
            "gold_social_scene_visible",
            lambda result: result.get("situated_social_scenario_visible"),
        ),
        "concrete_action_or_situated_speech_visible": (
            "gold_social_scene_visible",
            lambda result: result.get(
                "concrete_action_or_situated_speech_visible"
            ),
        ),
        "proposed_norm_plausibly_demonstrated": (
            "gold_label_matched_visible",
            lambda result: result.get("proposed_norm_plausibly_demonstrated"),
        ),
        "witnessed_violation_action_visible": (
            "gold_label_matched_visible",
            lambda result: result.get("witnessed_violation_action_visible"),
        ),
        "proposed_norm_and_organic": (
            "gold_label_matched_visible",
            lambda result: (
                "yes"
                if result.get("proposed_norm_plausibly_demonstrated") == "yes"
                and result.get("organic_vs_staged") == "organic"
                else "no"
            ),
        ),
        "proposed_norm_and_not_reaction_only": (
            "gold_label_matched_visible",
            lambda result: (
                "yes"
                if result.get("proposed_norm_plausibly_demonstrated") == "yes"
                and result.get("reaction_only_or_aftermath") == "no"
                else "no"
            ),
        ),
        "proposed_norm_organic_not_reaction_only": (
            "gold_label_matched_visible",
            lambda result: (
                "yes"
                if result.get("proposed_norm_plausibly_demonstrated") == "yes"
                and result.get("organic_vs_staged") == "organic"
                and result.get("reaction_only_or_aftermath") == "no"
                else "no"
            ),
        ),
    }
    report = {"parents": len(grouped), "fields": {}}
    for field, (target, answer) in fields.items():
        parents = []
        for parent, rows in grouped.items():
            usable = [
                (window, result)
                for window, result in rows
                if answer(result["result"]) in {"yes", "no"}
            ]
            if not usable:
                continue
            positives = sorted(
                window["window_index"]
                for window, result in usable
                if answer(result["result"]) == "yes"
            )
            exemplar = usable[0][1]
            parents.append(
                {
                    "parent_item_id": parent,
                    "gold": bool(exemplar[target]),
                    "positive_windows": len(positives),
                    "longest_positive_run": longest_run(positives),
                    "valid_windows": len(usable),
                }
            )
        rules = {}
        for count in (1, 2, 3):
            predictions = [
                parent["positive_windows"] >= count for parent in parents
            ]
            rules[f"at_least_{count}_positive_windows"] = metrics(
                [parent["gold"] for parent in parents], predictions
            )
        for count in (2, 3):
            predictions = [
                parent["longest_positive_run"] >= count for parent in parents
            ]
            rules[f"at_least_{count}_consecutive_windows"] = metrics(
                [parent["gold"] for parent in parents], predictions
            )
        report["fields"][field] = {
            "target": target,
            "rules": rules,
            "parents": parents,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
