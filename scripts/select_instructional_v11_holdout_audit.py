#!/usr/bin/env python3
"""Select the frozen V11 holdout candidates and disagreement controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_successful(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("error") is None and row.get("result") is not None:
            result[row["item_id"]] = row
    return result


def strict_vlm(result: dict[str, Any]) -> bool:
    return (
        result.get("exact_violation_demo") == "yes"
        and result.get("performed_violation_not_only_described") == "yes"
        and result.get("proposed_actor_behavior_target_match") == "yes"
        and result.get("event_polarity") == "violation"
    )


def blind_rank(seed: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{item_id}".encode()).hexdigest()


def select(
    source_path: Path,
    storyboard_path: Path,
    qwen_path: Path,
    glm_path: Path,
    consensus_path: Path,
    discovery_selection_path: Path,
    *,
    candidate_cap: int,
    control_count: int,
    blind_seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = load_jsonl(source_path)
    storyboards = {
        row["item_id"]: row for row in load_jsonl(storyboard_path)
    }
    qwen = latest_successful(qwen_path)
    glm = latest_successful(glm_path)
    consensus = latest_successful(consensus_path)
    item_ids = {row["item_id"] for row in source}
    if len(item_ids) != len(source):
        raise ValueError("source contains duplicate item_id")
    if len({row["uid"] for row in source}) != len(source):
        raise ValueError("source must contain one item per UID")
    for name, records in (
        ("storyboards", storyboards),
        ("qwen", qwen),
        ("glm", glm),
        ("consensus", consensus),
    ):
        if set(records) != item_ids:
            raise ValueError(
                f"{name} coverage mismatch: "
                f"missing={len(item_ids - set(records))}, "
                f"extra={len(set(records) - item_ids)}"
            )

    excluded_uids = {
        row["uid"] for row in load_jsonl(discovery_selection_path)
    }
    holdout = [row for row in source if row["uid"] not in excluded_uids]
    stage_by_item: dict[str, str] = {}
    candidates: list[dict[str, Any]] = []
    strata: dict[str, list[dict[str, Any]]] = {
        "qwen_only_exact": [],
        "glm_only_exact": [],
        "both_exact_consensus_reject": [],
        "remaining_non_candidate": [],
    }
    for row in holdout:
        item_id = row["item_id"]
        q_pass = strict_vlm(qwen[item_id]["result"])
        g_pass = strict_vlm(glm[item_id]["result"])
        c_pass = (
            consensus[item_id]["result"].get("strict_exact_violation") == "yes"
        )
        if q_pass and g_pass and c_pass:
            stage = "candidate"
            candidates.append(row)
        elif q_pass and not g_pass:
            stage = "qwen_only_exact"
            strata[stage].append(row)
        elif g_pass and not q_pass:
            stage = "glm_only_exact"
            strata[stage].append(row)
        elif q_pass and g_pass and not c_pass:
            stage = "both_exact_consensus_reject"
            strata[stage].append(row)
        else:
            stage = "remaining_non_candidate"
            strata[stage].append(row)
        stage_by_item[item_id] = stage

    selected_candidates = candidates[:candidate_cap]
    controls: list[dict[str, Any]] = []
    for name in (
        "qwen_only_exact",
        "glm_only_exact",
        "both_exact_consensus_reject",
    ):
        controls.extend(strata[name][:10])
    if len(controls) < control_count:
        controls.extend(
            strata["remaining_non_candidate"][: control_count - len(controls)]
        )
    controls = controls[:control_count]
    if len(controls) != control_count:
        raise ValueError("not enough holdout controls")

    selected = [
        (row, "primary_candidate") for row in selected_candidates
    ] + [(row, "control_reject") for row in controls]
    selected.sort(key=lambda item: blind_rank(blind_seed, item[0]["item_id"]))
    semantic: list[dict[str, Any]] = []
    blind: list[dict[str, Any]] = []
    for audit_index, (row, band) in enumerate(selected):
        item_id = row["item_id"]
        storyboard = storyboards[item_id]
        semantic.append(
            {
                **row,
                "audit_index": audit_index,
                "band": band,
                "candidate_stage": stage_by_item[item_id],
                "storyboard": storyboard,
                "qwen_v11": qwen[item_id],
                "glm_v11": glm[item_id],
                "consensus_v11": consensus[item_id],
            }
        )
        blind.append(
            {
                "audit_index": audit_index,
                "candidate_id": f"v11holdout-{audit_index:04d}",
                "sheet_path": storyboard["sheet_path"],
                "sheet_sha256": storyboard["sheet_sha256"],
            }
        )
    summary = {
        "source_items": len(source),
        "excluded_discovery_uids": len(excluded_uids),
        "holdout_items": len(holdout),
        "candidate_population_items": len(candidates),
        "candidate_population_uids": len({row["uid"] for row in candidates}),
        "candidate_selected_items": len(selected_candidates),
        "candidate_cap": candidate_cap,
        "control_items": len(controls),
        "selected_items": len(selected),
        "holdout_stage_counts": dict(
            sorted(Counter(stage_by_item.values()).items())
        ),
        "control_stage_counts": dict(
            sorted(
                Counter(stage_by_item[row["item_id"]] for row in controls).items()
            )
        ),
        "minimum_candidate_count_met": len(selected_candidates) >= 30,
        "minimum_candidate_uid_count_met": (
            len({row["uid"] for row in selected_candidates}) >= 20
        ),
        "blind_seed": blind_seed,
        "semantic_metadata_in_blind_manifest": False,
        "corpus_mutated": False,
    }
    return semantic, blind, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--storyboards", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--glm", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--discovery-selection", type=Path, required=True)
    parser.add_argument("--candidate-cap", type=int, default=60)
    parser.add_argument("--control-count", type=int, default=30)
    parser.add_argument(
        "--blind-seed", default="20260728-v11-holdout-audit-v1"
    )
    parser.add_argument("--out-semantic", type=Path, required=True)
    parser.add_argument("--out-blind", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.out_semantic, args.out_blind, args.out_summary):
        if path.exists():
            raise SystemExit(f"refusing to overwrite: {path}")
    semantic, blind, summary = select(
        args.source,
        args.storyboards,
        args.qwen,
        args.glm,
        args.consensus,
        args.discovery_selection,
        candidate_cap=args.candidate_cap,
        control_count=args.control_count,
        blind_seed=args.blind_seed,
    )
    args.out_semantic.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in semantic)
    )
    args.out_blind.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind)
    )
    summary["semantic_selection_sha256"] = sha256(args.out_semantic)
    summary["blind_manifest_sha256"] = sha256(args.out_blind)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
