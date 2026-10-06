#!/usr/bin/env python3
"""Select the frozen V12 high-confidence holdout and deterministic controls."""

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


def latest_successful(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("error") is None and isinstance(row.get("result"), dict):
            rows[row["item_id"]] = row
    return rows


def strict_metadata_vlm(record: dict[str, Any]) -> bool:
    result = record["result"]
    return (
        result.get("exact_violation_demo") == "yes"
        and result.get("performed_violation_not_only_described") == "yes"
        and result.get("proposed_actor_behavior_target_match") == "yes"
        and result.get("event_polarity") == "violation"
    )


def opaque_rank(seed: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{item_id}".encode()).hexdigest()


def select(
    source_path: Path,
    storyboard_path: Path,
    v12_path: Path,
    qwen_path: Path,
    gemma_metadata_path: Path,
    exclusion_paths: list[Path],
    *,
    candidate_cap: int,
    control_per_stratum: int,
    blind_seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = load_jsonl(source_path)
    storyboards = {
        row["item_id"]: row for row in load_jsonl(storyboard_path)
    }
    v12 = latest_successful(v12_path)
    qwen = latest_successful(qwen_path)
    gemma = latest_successful(gemma_metadata_path)
    item_ids = {row["item_id"] for row in source}
    if len(item_ids) != len(source):
        raise ValueError("source contains duplicate item_id")
    if len({row["uid"] for row in source}) != len(source):
        raise ValueError("source must contain one item per UID")
    for name, records in (
        ("storyboards", storyboards),
        ("v12", v12),
        ("qwen", qwen),
        ("gemma_metadata", gemma),
    ):
        if set(records) != item_ids:
            raise ValueError(
                f"{name} coverage mismatch: "
                f"missing={len(item_ids - set(records))}, "
                f"extra={len(set(records) - item_ids)}"
            )

    excluded_uids: set[str] = set()
    for path in exclusion_paths:
        excluded_uids.update(row["uid"] for row in load_jsonl(path))
    holdout = [row for row in source if row["uid"] not in excluded_uids]
    rejected = [
        row
        for row in holdout
        if v12[row["item_id"]]["result"].get("strict_exact_alignment")
        != "yes"
    ]
    candidates = [
        row
        for row in holdout
        if v12[row["item_id"]]["result"].get("strict_exact_alignment")
        == "yes"
    ]
    selected_candidates = candidates[:candidate_cap]

    controls: list[tuple[dict[str, Any], str]] = []
    used: set[str] = set()

    def take(rows: list[dict[str, Any]], name: str, count: int) -> None:
        for row in rows:
            if len([1 for _, band in controls if band == name]) >= count:
                break
            if row["item_id"] in used:
                continue
            used.add(row["item_id"])
            controls.append((row, name))

    take(
        [
            row
            for row in rejected
            if strict_metadata_vlm(qwen[row["item_id"]])
        ],
        "control_qwen_exact_v12_reject",
        control_per_stratum,
    )
    take(
        [
            row
            for row in rejected
            if strict_metadata_vlm(gemma[row["item_id"]])
        ],
        "control_gemma_exact_v12_reject",
        control_per_stratum,
    )
    take(
        rejected,
        "control_remaining_v12_reject",
        control_per_stratum,
    )
    control_target = 3 * control_per_stratum
    if len(controls) < control_target:
        for row in rejected:
            if len(controls) >= control_target:
                break
            if row["item_id"] in used:
                continue
            used.add(row["item_id"])
            controls.append((row, "control_fill_v12_reject"))
    if len(controls) != control_target:
        raise ValueError("not enough distinct V12-rejected controls")

    selected = [
        (row, "primary_v12_candidate") for row in selected_candidates
    ] + controls
    selected.sort(key=lambda pair: opaque_rank(blind_seed, pair[0]["item_id"]))
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
                "storyboard": storyboard,
                "v12_causal": v12[item_id],
                "qwen_metadata": qwen[item_id],
                "gemma_metadata": gemma[item_id],
            }
        )
        blind.append(
            {
                "audit_index": audit_index,
                "candidate_id": f"v12holdout-{audit_index:04d}",
                "sheet_path": storyboard["sheet_path"],
                "sheet_sha256": storyboard["sheet_sha256"],
            }
        )

    control_counts = Counter(band for _, band in controls)
    summary = {
        "source_items": len(source),
        "excluded_discovery_uids": len(excluded_uids),
        "holdout_items": len(holdout),
        "candidate_population_items": len(candidates),
        "candidate_population_uids": len({row["uid"] for row in candidates}),
        "candidate_selected_items": len(selected_candidates),
        "candidate_selected_uids": len(
            {row["uid"] for row in selected_candidates}
        ),
        "candidate_cap": candidate_cap,
        "control_items": len(controls),
        "control_counts": dict(sorted(control_counts.items())),
        "selected_items": len(selected),
        "minimum_candidate_count_met": len(selected_candidates) >= 30,
        "minimum_candidate_uid_count_met": (
            len({row["uid"] for row in selected_candidates}) >= 20
        ),
        "blind_seed": blind_seed,
        "semantic_metadata_in_blind_manifest": False,
        "corpus_mutated": False,
    }
    return semantic, blind, summary


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--storyboards", type=Path, required=True)
    parser.add_argument("--v12", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gemma-metadata", type=Path, required=True)
    parser.add_argument(
        "--exclude-selection", type=Path, action="append", required=True
    )
    parser.add_argument("--candidate-cap", type=int, default=60)
    parser.add_argument("--control-per-stratum", type=int, default=10)
    parser.add_argument(
        "--blind-seed", default="20260728-v12-holdout-audit-v1"
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
        args.v12,
        args.qwen,
        args.gemma_metadata,
        args.exclude_selection,
        candidate_cap=args.candidate_cap,
        control_per_stratum=args.control_per_stratum,
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
