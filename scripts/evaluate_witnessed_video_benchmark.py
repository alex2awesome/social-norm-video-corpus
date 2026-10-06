#!/usr/bin/env python3
"""Evaluate witnessed full-video VLM gates against frozen manual dispositions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


STAGED_DEPICTIONS = {
    "enacted_scene",
    "film_tv",
    "animation",
    "puppet_or_toy",
    "screen_scenario",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def atomic_common(result: dict[str, Any]) -> bool:
    return (
        result.get("social_norm_domain") == "yes"
        and result.get("situated_social_scenario_visible") == "yes"
        and result.get("observable_social_behavior_or_speech") == "yes"
        and result.get("usable_demo_after_relabel") == "yes"
        and result.get("procedural_or_nonsocial_activity_only") == "no"
        and result.get("presentation_or_context_only") == "no"
    )


def predict(
    row: dict[str, Any],
    route: str,
    *,
    allow_broad: bool,
) -> bool | None:
    if row.get("error") or not isinstance(row.get("result"), dict):
        return None
    result = row["result"]
    allowed_localization = {"clean", "broad"} if allow_broad else {"clean"}
    if result.get("localization_quality") not in allowed_localization:
        return False
    common = atomic_common(result)
    witnessed = (
        common
        and result.get("witnessed_action_then_reaction_organic") == "yes"
    )
    instructional = (
        common
        and result.get("depiction_type") in STAGED_DEPICTIONS
        and result.get("witnessed_action_then_reaction_organic") != "yes"
    )
    if route == "strict_current":
        return witnessed and result.get("proposed_norm_supported") == "yes"
    if route == "witnessed_recovery":
        return witnessed
    if route == "instructional_reroute":
        return instructional
    if route == "any_visual_recovery":
        return witnessed or instructional
    raise ValueError(f"unsupported route: {route}")


def metrics(
    gold: dict[str, bool],
    predictions: dict[str, bool | None],
) -> dict[str, Any]:
    evaluated = [item_id for item_id in gold if predictions.get(item_id) is not None]
    selected = [item_id for item_id in evaluated if predictions[item_id] is True]
    tp_items = [item_id for item_id in selected if gold[item_id]]
    fp_items = [item_id for item_id in selected if not gold[item_id]]
    fn_items = [
        item_id
        for item_id in evaluated
        if predictions[item_id] is False and gold[item_id]
    ]
    tn_items = [
        item_id
        for item_id in evaluated
        if predictions[item_id] is False and not gold[item_id]
    ]
    return {
        "gold_positive": sum(gold.values()),
        "evaluated": len(evaluated),
        "abstained_or_error": len(gold) - len(evaluated),
        "predicted_positive": len(selected),
        "true_positive": len(tp_items),
        "false_positive": len(fp_items),
        "false_negative": len(fn_items),
        "true_negative": len(tn_items),
        "precision": len(tp_items) / len(selected) if selected else None,
        "recall": (
            len(tp_items) / sum(gold[item_id] for item_id in evaluated)
            if any(gold[item_id] for item_id in evaluated)
            else None
        ),
        "predicted_positive_items": selected,
        "true_positive_items": tp_items,
        "false_positive_items": fp_items,
        "false_negative_items": fn_items,
        "true_negative_items": tn_items,
    }


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

    gold_fields: dict[str, str] = {
        "strict_current": "gold_strict_current",
        "witnessed_recovery": "gold_witnessed_recovery",
        "instructional_reroute": "gold_instructional_reroute",
        "any_visual_recovery": "gold_any_visual_recovery",
    }
    reports = {}
    for allow_broad in (False, True):
        band = "clean_or_broad" if allow_broad else "clean"
        for route, gold_field in gold_fields.items():
            gold = {
                item_id: bool(row[gold_field])
                for item_id, row in benchmark.items()
            }
            first = {
                item_id: predict(primary[item_id], route, allow_broad=allow_broad)
                for item_id in benchmark
            }
            second = {
                item_id: predict(secondary[item_id], route, allow_broad=allow_broad)
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
