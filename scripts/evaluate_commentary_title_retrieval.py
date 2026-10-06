#!/usr/bin/env python3
"""Evaluate title-conditioned commentary retrieval against the frozen ledger.

This evaluates discovery/localization signals only.  It intentionally reports
separate visual-event, exact-label, and any-usable targets because situated
speech and instructional reroutes should not be conflated with pixel-visible
physical actions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_ledger(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = {
            int(row["audit_index"]): row
            for row in csv.DictReader(handle, delimiter="\t")
        }
    if not rows:
        raise ValueError("manual ledger is empty")
    return rows


def successful_results(path: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["audit_index"]): row["result"]
        for row in read_jsonl(path)
        if row.get("error") is None and isinstance(row.get("result"), dict)
    }


def has_deixis_window(path: Path) -> set[int]:
    return {
        int(row["audit_index"])
        for row in read_jsonl(path)
        if int(row.get("visual_deixis_window_count", 0)) > 0
    }


def candidate_yes(result: dict[str, Any] | None) -> bool:
    return bool(result) and result.get("candidate_event") == "yes"


def strict_literal_yes(result: dict[str, Any] | None) -> bool:
    return bool(result) and all(
        result.get(field) == "yes"
        for field in (
            "candidate_event",
            "title_action_visible",
            "actor_action_target_same_event",
        )
    )


def metrics(
    gold: set[int],
    selected: set[int],
    universe: set[int],
) -> dict[str, Any]:
    selected = selected & universe
    tp = sorted(gold & selected)
    fp = sorted(selected - gold)
    fn = sorted(gold - selected)
    return {
        "selected": len(selected),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": len(tp) / len(selected) if selected else None,
        "recall": len(tp) / len(gold) if gold else None,
        "tp_indices": tp,
        "fp_indices": fp,
        "fn_indices": fn,
    }


def evaluate(
    ledger: dict[int, dict[str, str]],
    qwen: dict[int, dict[str, Any]],
    gemma: dict[int, dict[str, Any]],
    deixis: set[int],
) -> dict[str, Any]:
    universe = set(ledger)
    visual_gold = {
        index
        for index, row in ledger.items()
        if row["manual_class"].startswith("pass_visual")
    }
    exact_gold = {
        index
        for index, row in ledger.items()
        if row["usable_weak_supervision"] == "yes"
        and row["label_alignment"] == "exact"
    }
    usable_gold = {
        index
        for index, row in ledger.items()
        if row["usable_weak_supervision"] == "yes"
    }

    qwen_yes = {index for index in universe if candidate_yes(qwen.get(index))}
    gemma_yes = {index for index in universe if candidate_yes(gemma.get(index))}
    qwen_strict = {
        index for index in universe if strict_literal_yes(qwen.get(index))
    }
    gemma_strict = {
        index for index in universe if strict_literal_yes(gemma.get(index))
    }
    selections = {
        "qwen_candidate_yes": qwen_yes,
        "gemma_candidate_yes": gemma_yes,
        "dual_candidate_yes": qwen_yes & gemma_yes,
        "either_candidate_yes": qwen_yes | gemma_yes,
        "qwen_strict_literal": qwen_strict,
        "gemma_strict_literal": gemma_strict,
        "dual_strict_literal": qwen_strict & gemma_strict,
        "transcript_visual_deixis": deixis,
        "dual_candidate_or_deixis": (qwen_yes & gemma_yes) | deixis,
        "either_candidate_and_deixis": (qwen_yes | gemma_yes) & deixis,
    }
    targets = {
        "pixel_visible_event": visual_gold,
        "usable_exact_label": exact_gold,
        "usable_any_tier": usable_gold,
    }
    return {
        "kind": "commentary_title_retrieval_shadow_evaluation",
        "policy": "discovery_only_not_a_keep_filter",
        "items": len(universe),
        "coverage": {
            "qwen": len(universe & set(qwen)),
            "gemma": len(universe & set(gemma)),
            "deixis_rows": len(universe & deixis),
        },
        "gold_counts": {name: len(rows) for name, rows in targets.items()},
        "rules": {
            rule_name: {
                target_name: metrics(gold, selected, universe)
                for target_name, gold in targets.items()
            }
            for rule_name, selected in selections.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--deixis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    report = evaluate(
        read_ledger(args.ledger),
        successful_results(args.qwen),
        successful_results(args.gemma),
        has_deixis_window(args.deixis),
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
