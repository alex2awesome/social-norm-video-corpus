#!/usr/bin/env python3
"""Evaluate frozen open-VLM outputs on the completed commentary source ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def result(row: dict | None) -> dict:
    return (row or {}).get("result") or {}


def yes(value: object) -> bool:
    return str(value).lower() == "yes"


def retrieval_core(row: dict | None) -> bool:
    r = result(row)
    return all(
        yes(r.get(key))
        for key in (
            "candidate_event",
            "title_action_visible",
            "title_actor_visible",
            "title_target_or_property_visible",
            "actor_action_target_same_event",
        )
    )


def retrieval_exact(row: dict | None) -> bool:
    r = result(row)
    return retrieval_core(row) and r.get("title_alignment") == "exact"


def retrieval_exact_no_followup(row: dict | None) -> bool:
    r = result(row)
    return retrieval_exact(row) and r.get("followup") == "none"


def labelblind_core(row: dict | None) -> bool:
    r = result(row)
    return all(
        yes(r.get(key))
        for key in (
            "performed_social_behavior",
            "actor_action_target_same_event",
            "target_behavior_performed_not_described",
            "socially_evaluable_without_metadata",
            "demo_usable",
        )
    )


def metrics(indices: list[int], selected: set[int], gold: set[int]) -> dict:
    universe = set(indices)
    selected = selected & universe
    gold = gold & universe
    tp = selected & gold
    fp = selected - gold
    fn = gold - selected
    return {
        "selected": len(selected),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": len(tp) / len(selected) if selected else None,
        "recall": len(tp) / len(gold) if gold else None,
        "selected_indices": sorted(selected),
        "fp_indices": sorted(fp),
    }


def evaluate(
    manual_path: Path,
    qwen_blind_path: Path,
    qwen_retrieval_path: Path,
    gemma_retrieval_path: Path,
) -> dict:
    manual_rows = read_jsonl(manual_path)
    manual = {int(row["audit_index"]): row for row in manual_rows}
    qblind = {int(row["audit_index"]): row for row in read_jsonl(qwen_blind_path)}
    qret = {int(row["audit_index"]): row for row in read_jsonl(qwen_retrieval_path)}
    gret = {int(row["audit_index"]): row for row in read_jsonl(gemma_retrieval_path)}
    expected = set(range(154))
    if set(manual) != expected or set(qblind) != expected or set(qret) != expected:
        raise ValueError("Manual, Qwen-blind, and Qwen-retrieval inputs must cover 0..153")
    if set(gret) != expected:
        raise ValueError("Gemma retrieval input must contain all 154 rows, including errors")

    rules: dict[str, Callable[[int], bool]] = {
        "qwen_labelblind_core": lambda i: labelblind_core(qblind[i]),
        "qwen_retrieval_core": lambda i: retrieval_core(qret[i]),
        "qwen_retrieval_exact": lambda i: retrieval_exact(qret[i]),
        "qwen_retrieval_exact_no_followup": lambda i: retrieval_exact_no_followup(
            qret[i]
        ),
        "gemma_retrieval_core": lambda i: retrieval_core(gret[i]),
        "gemma_retrieval_exact": lambda i: retrieval_exact(gret[i]),
        "gemma_retrieval_exact_no_followup": lambda i: retrieval_exact_no_followup(
            gret[i]
        ),
        "dual_retrieval_core": lambda i: retrieval_core(qret[i])
        and retrieval_core(gret[i]),
        "dual_retrieval_exact": lambda i: retrieval_exact(qret[i])
        and retrieval_exact(gret[i]),
        "either_exact_and_labelblind": lambda i: (
            retrieval_exact(qret[i]) or retrieval_exact(gret[i])
        )
        and labelblind_core(qblind[i]),
        "dual_core_and_labelblind": lambda i: retrieval_core(qret[i])
        and retrieval_core(gret[i])
        and labelblind_core(qblind[i]),
    }
    gold_any = {
        i for i, row in manual.items() if bool(row["source_usable_candidate"])
    }
    gold_exact = {
        i
        for i, row in manual.items()
        if row["route"] == "exact_visual_localization_review"
    }
    slices = {
        "all_154": list(range(154)),
        "clear_94_validation": [
            i
            for i, row in manual.items()
            if row["adjudication_provenance"]
            == "clear_source_post_reveal_adjudication"
        ],
        "ambiguous_60_development": [
            i
            for i, row in manual.items()
            if row["adjudication_provenance"] == "moving_video_post_reveal_ledger"
        ],
    }
    output = {
        "kind": "commentary_capture_title_open_vlm_full_manual_evaluation",
        "policy": "shadow_ranking_only_no_automatic_acceptance",
        "gold_counts": {
            "source_usable_any_route": len(gold_any),
            "exact_title_event": len(gold_exact),
            "routes": dict(Counter(row["route"] for row in manual.values())),
        },
        "slices": {},
    }
    for slice_name, indices in slices.items():
        slice_result = {"items": len(indices), "rules": {}}
        for rule_name, predicate in rules.items():
            selected = {i for i in indices if predicate(i)}
            slice_result["rules"][rule_name] = {
                "usable_any_route": metrics(indices, selected, gold_any),
                "exact_title_event": metrics(indices, selected, gold_exact),
            }
        output["slices"][slice_name] = slice_result
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--qwen-labelblind", type=Path, required=True)
    parser.add_argument("--qwen-retrieval", type=Path, required=True)
    parser.add_argument("--gemma-retrieval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = evaluate(
        args.manual,
        args.qwen_labelblind,
        args.qwen_retrieval,
        args.gemma_retrieval,
    )
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
