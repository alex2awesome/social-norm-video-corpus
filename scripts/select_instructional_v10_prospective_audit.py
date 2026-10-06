#!/usr/bin/env python3
"""Freeze every V10 candidate plus deterministic rejects for blind review."""

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def latest_successful(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("error") is None and row.get("result") is not None:
            result[row["item_id"]] = row
    return result


def strict_v9a(result: dict[str, Any]) -> bool:
    return (
        result.get("observable_event") == "yes"
        and result.get("same_event_actor_action_target") == "yes"
        and result.get("event_temporally_localized") == "yes"
        and result.get("scene_role") == "demonstrated_event"
    )


def stage(
    row: dict[str, Any],
    v9a: dict[str, dict[str, Any]],
    v9b: dict[str, dict[str, Any]],
    v10a: dict[str, dict[str, Any]],
    v10c: dict[str, dict[str, Any]],
) -> str:
    item_id = row["item_id"]
    if row.get("source_platform") != "youtube":
        return "platform_reject"
    if row.get("polarity") != "violation":
        return "polarity_reject"
    if not strict_v9a(v9a[item_id]["result"]):
        return "v9a_reject"
    if v9b[item_id]["result"].get("usable_after_relabel") != "yes":
        return "v9b_reject"
    if v10a[item_id]["result"].get("demo_usable") != "yes":
        return "v10a_reject"
    if v10c[item_id]["result"].get("strict_exact_candidate") != "yes":
        return "v10c_reject"
    return "candidate"


def select(
    source_path: Path,
    storyboard_path: Path,
    v9a_path: Path,
    v9b_path: Path,
    v10a_path: Path,
    v10c_path: Path,
    *,
    control_count: int,
    control_seed: str,
    blind_seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = load_jsonl(source_path)
    storyboards = {
        row["item_id"]: row for row in load_jsonl(storyboard_path)
    }
    v9a = latest_successful(v9a_path)
    v9b = latest_successful(v9b_path)
    v10a = latest_successful(v10a_path)
    v10c = latest_successful(v10c_path)
    item_ids = {row["item_id"] for row in source}
    if len(source) != len(item_ids):
        raise ValueError("source contains duplicate item_id")
    if len({row["uid"] for row in source}) != len(source):
        raise ValueError("source must contain one clip per UID")
    for name, rows in [
        ("storyboard", storyboards),
        ("v9a", v9a),
        ("v9b", v9b),
        ("v10a", v10a),
        ("v10c", v10c),
    ]:
        if set(rows) != item_ids:
            raise ValueError(
                f"{name} coverage mismatch: missing={len(item_ids - set(rows))}, "
                f"extra={len(set(rows) - item_ids)}"
            )

    staged = [(row, stage(row, v9a, v9b, v10a, v10c)) for row in source]
    candidates = [row for row, value in staged if value == "candidate"]
    rejects = [row for row, value in staged if value != "candidate"]
    rejects.sort(key=lambda row: rank(control_seed, row["item_id"]))
    controls = rejects[:control_count]
    if len(controls) != control_count:
        raise ValueError("not enough rejects for requested controls")

    stage_by_item = {row["item_id"]: value for row, value in staged}
    selected = [
        (row, "primary_candidate")
        for row in candidates
    ] + [
        (row, "control_reject")
        for row in controls
    ]
    selected.sort(key=lambda value: rank(blind_seed, value[0]["item_id"]))
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
                "v9a": v9a[item_id],
                "v9b": v9b[item_id],
                "v10a": v10a[item_id],
                "v10c": v10c[item_id],
            }
        )
        blind.append(
            {
                "audit_index": audit_index,
                "candidate_id": f"v10prospective-{audit_index:04d}",
                "sheet_path": storyboard["sheet_path"],
                "sheet_sha256": storyboard["sheet_sha256"],
            }
        )
    summary = {
        "source_items": len(source),
        "source_uids": len({row["uid"] for row in source}),
        "candidate_items": len(candidates),
        "candidate_uids": len({row["uid"] for row in candidates}),
        "control_items": len(controls),
        "selected_items": len(selected),
        "source_stage_counts": dict(
            sorted(Counter(value for _, value in staged).items())
        ),
        "control_stage_counts": dict(
            sorted(Counter(stage_by_item[row["item_id"]] for row in controls).items())
        ),
        "minimum_candidate_count_met": len(candidates) >= 30,
        "control_seed": control_seed,
        "blind_seed": blind_seed,
        "semantic_metadata_in_blind_manifest": False,
        "corpus_mutated": False,
    }
    return semantic, blind, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--storyboards", type=Path, required=True)
    parser.add_argument("--v9a", type=Path, required=True)
    parser.add_argument("--v9b", type=Path, required=True)
    parser.add_argument("--v10a", type=Path, required=True)
    parser.add_argument("--v10c", type=Path, required=True)
    parser.add_argument("--control-count", type=int, default=30)
    parser.add_argument(
        "--control-seed",
        default="20260728-v10-prospective-controls-v1",
    )
    parser.add_argument(
        "--blind-seed",
        default="20260728-v10-prospective-blind-v1",
    )
    parser.add_argument("--out-semantic", type=Path, required=True)
    parser.add_argument("--out-blind", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()
    for path in [args.out_semantic, args.out_blind, args.out_summary]:
        if path.exists():
            raise SystemExit(f"refusing to overwrite: {path}")
    semantic, blind, summary = select(
        args.source,
        args.storyboards,
        args.v9a,
        args.v9b,
        args.v10a,
        args.v10c,
        control_count=args.control_count,
        control_seed=args.control_seed,
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
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
