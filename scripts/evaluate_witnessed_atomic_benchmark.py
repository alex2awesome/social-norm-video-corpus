#!/usr/bin/env python3
"""Evaluate deterministic witnessed rules over atomic video-VLM observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_witnessed_video_benchmark import load_jsonl, metrics
else:
    from evaluate_witnessed_video_benchmark import load_jsonl, metrics


SOCIAL_EXPECTATIONS = {"interpersonal_treatment", "shared_public_conduct"}
REACTION_ROLES = {"bystander", "authority_or_host", "organic_audience"}
NORMATIVE_REACTIONS = {
    "targeted_objection",
    "correction_or_sanction",
    "protective_intervention",
}
ORGANIC = {"organic", "hidden_camera_genuine"}
STAGED = {"scripted", "animation"}
CLEAR_DEMO = {"clear_visual", "clear_audiovisual"}


def witnessed_atomic(result: dict[str, Any], *, require_bounds: bool) -> bool:
    supported = (
        result.get("action_visible") == "yes"
        and result.get("action_voluntary") == "yes"
        and result.get("expectation_kind") in SOCIAL_EXPECTATIONS
        and result.get("reaction_visible_or_audibly_grounded") == "yes"
        and result.get("reaction_source_role") in REACTION_ROLES
        and result.get("reaction_content") in NORMATIVE_REACTIONS
        and result.get("action_established_before_reaction") == "yes"
        and result.get("reaction_targets_action") == "yes"
        and result.get("authenticity") in ORGANIC
        and result.get("pre_reaction_demo_quality") in CLEAR_DEMO
    )
    if not supported or not require_bounds:
        return supported
    action_end = result.get("action_end_percent", -1)
    reaction_start = result.get("reaction_start_percent", -1)
    return (
        isinstance(action_end, (int, float))
        and isinstance(reaction_start, (int, float))
        and 0 < action_end <= reaction_start <= 100
    )


def instructional_atomic(result: dict[str, Any]) -> bool:
    return (
        result.get("action_visible") == "yes"
        and result.get("action_voluntary") == "yes"
        and result.get("expectation_kind") in SOCIAL_EXPECTATIONS
        and result.get("authenticity") in STAGED
        and result.get("pre_reaction_demo_quality") in CLEAR_DEMO
        and result.get("proposed_label_relation") in {"exact", "repairable"}
    )


def predict(
    row: dict[str, Any],
    route: str,
    *,
    require_bounds: bool,
) -> bool | None:
    if row.get("error") or not isinstance(row.get("result"), dict):
        return None
    result = row["result"]
    witnessed = witnessed_atomic(result, require_bounds=require_bounds)
    instructional = instructional_atomic(result)
    if route == "strict_current":
        return witnessed and result.get("proposed_label_relation") == "exact"
    if route == "witnessed_recovery":
        return witnessed
    if route == "instructional_reroute":
        return instructional
    if route == "any_visual_recovery":
        return witnessed or instructional
    raise ValueError(f"unsupported route: {route}")


def evaluate(
    benchmark_rows: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
    secondary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    benchmark = {row["item_id"]: row for row in benchmark_rows}
    primary = {row["item_id"]: row for row in primary_rows}
    secondary = {row["item_id"]: row for row in secondary_rows}
    if any(
        len(mapping) != len(rows)
        for mapping, rows in (
            (benchmark, benchmark_rows),
            (primary, primary_rows),
            (secondary, secondary_rows),
        )
    ):
        raise ValueError("benchmark and model inputs require unique item IDs")
    if set(primary) != set(benchmark) or set(secondary) != set(benchmark):
        raise ValueError("benchmark and model inputs must cover the same items")

    gold_fields = {
        "strict_current": "gold_strict_current",
        "witnessed_recovery": "gold_witnessed_recovery",
        "instructional_reroute": "gold_instructional_reroute",
        "any_visual_recovery": "gold_any_visual_recovery",
    }
    reports = {}
    for require_bounds in (False, True):
        band = "exact_bounds" if require_bounds else "review_triage"
        for route, gold_field in gold_fields.items():
            gold = {
                item_id: bool(row[gold_field])
                for item_id, row in benchmark.items()
            }
            first = {
                item_id: predict(
                    primary[item_id], route, require_bounds=require_bounds
                )
                for item_id in benchmark
            }
            second = {
                item_id: predict(
                    secondary[item_id], route, require_bounds=require_bounds
                )
                for item_id in benchmark
            }
            intersection = {
                item_id: (
                    None
                    if first[item_id] is None or second[item_id] is None
                    else first[item_id] and second[item_id]
                )
                for item_id in benchmark
            }
            reports[f"{route}_{band}"] = {
                "primary": metrics(gold, first),
                "secondary": metrics(gold, second),
                "intersection": metrics(gold, intersection),
            }
    return {
        "status": "shadow_only_do_not_promote_without_source_disjoint_replication",
        "items": len(benchmark_rows),
        "rules": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        load_jsonl(args.benchmark),
        load_jsonl(args.primary),
        load_jsonl(args.secondary),
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        name: models["intersection"]
        for name, models in report["rules"].items()
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
