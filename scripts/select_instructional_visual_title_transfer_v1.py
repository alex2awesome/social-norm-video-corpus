#!/usr/bin/env python3
"""Freeze a stratified transfer for visual-consensus plus scene-title cues."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_instructional_scene_title_cues import title_cues
else:
    from evaluate_instructional_scene_title_cues import title_cues


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def stratum(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("category") or "unknown"),
        str(row.get("polarity") or "unknown"),
    )


def one_demo_per_source(
    rows: list[dict[str, Any]], seed: str, excluded_uids: set[str]
) -> list[dict[str, Any]]:
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_items: set[str] = set()
    for row in rows:
        item_id = str(row.get("item_id") or "")
        uid = str(row.get("uid") or "")
        if not item_id or not uid or not row.get("source_clip"):
            continue
        if item_id in seen_items:
            raise ValueError(f"duplicate item_id: {item_id}")
        seen_items.add(item_id)
        if uid not in excluded_uids:
            by_uid[uid].append(row)
    output = []
    for uid, values in by_uid.items():
        values.sort(key=lambda row: stable_key(f"{seed}:demo", str(row["item_id"])))
        chosen = dict(values[0])
        chosen["scene_title_candidate"] = bool(
            title_cues(str(chosen.get("title") or ""))["scene_title_candidate"]
        )
        output.append(chosen)
    return output


def balanced_select(
    rows: list[dict[str, Any]], count: int, seed: str
) -> list[dict[str, Any]]:
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pools[stratum(row)].append(row)
    for key, values in pools.items():
        values.sort(key=lambda row: stable_key(f"{seed}:{key}", str(row["item_id"])))
    strata = sorted(pools, key=lambda key: stable_key(f"{seed}:strata", repr(key)))
    offsets = {key: 0 for key in strata}
    selected = []
    while len(selected) < count:
        progress = False
        for key in strata:
            if offsets[key] < len(pools[key]):
                selected.append(pools[key][offsets[key]])
                offsets[key] += 1
                progress = True
            if len(selected) == count:
                break
        if not progress:
            raise ValueError("insufficient rows for requested stratum")
    return selected


def select(
    rows: list[dict[str, Any]],
    title_positive: int,
    title_negative: int,
    seed: str,
    excluded_uids: set[str],
) -> list[dict[str, Any]]:
    unique = one_demo_per_source(rows, seed, excluded_uids)
    positive = balanced_select(
        [row for row in unique if row["scene_title_candidate"]],
        title_positive, f"{seed}:title-positive",
    )
    negative = balanced_select(
        [row for row in unique if not row["scene_title_candidate"]],
        title_negative, f"{seed}:title-negative",
    )
    combined = positive + negative
    if len({row["uid"] for row in combined}) != len(combined):
        raise AssertionError("selection is not source-disjoint")
    return sorted(
        combined, key=lambda row: stable_key(f"{seed}:blind", str(row["item_id"]))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path, action="append", required=True)
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title-positive", type=int, default=70)
    parser.add_argument("--title-negative", type=int, default=30)
    parser.add_argument("--seed", default="instructional-visual-title-transfer-v1")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    excluded = {
        line.strip()
        for path in args.exclude_uids
        for line in path.read_text().splitlines()
        if line.strip()
    }
    excluded.update(
        str(row["uid"])
        for path in args.exclude_jsonl
        for row in read_jsonl(path)
        if row.get("uid")
    )
    selected = select(
        read_jsonl(args.source), args.title_positive, args.title_negative,
        args.seed, excluded,
    )
    args.out.mkdir(parents=True)
    semantic = []
    blind = []
    for index, row in enumerate(selected):
        candidate_id = f"visual-title-transfer-{index:04d}"
        semantic.append({**row, "audit_index": index, "candidate_id": candidate_id})
        blind.append({
            "audit_index": index,
            "candidate_id": candidate_id,
            "item_id": row["item_id"],
            "uid": row["uid"],
            "pillar": "instructional",
            "source_clip": row["source_clip"],
        })
    semantic_path = args.out / "sealed_selection.jsonl"
    blind_path = args.out / "blind_source_manifest.jsonl"
    semantic_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in semantic))
    blind_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in blind))
    counts = Counter(bool(row["scene_title_candidate"]) for row in selected)
    contract = {
        "kind": "instructional_visual_title_conjunction_transfer_v1",
        "status": "preregistered_blind_audit_pending",
        "seed": args.seed,
        "items": len(selected),
        "unique_uids": len({row["uid"] for row in selected}),
        "stratified_title_positive": counts[True],
        "stratified_title_negative_control": counts[False],
        "frozen_rule": {
            "all": ["qwen_v22_demo_pass_yes", "gemma_v22_demo_pass_yes", "scene_title_candidate"],
            "target": "manual_visual_demo_any_valid_medium",
        },
        "manual_requirements": {
            "blind_storyboard_review_every_item": True,
            "manual_review_every_successful_model_output": True,
            "post_reveal_original_label_alignment_every_item": True,
        },
        "gate": {
            "minimum_rendered_items": 95,
            "minimum_selected": 30,
            "minimum_visual_demo_precision": 0.90,
            "minimum_visual_demo_precision_wilson_95_lower": 0.75,
            "recall_reported_on_negative_controls": True,
            "permitted_if_passed": "candidate_generation_and_manual_review_ranking_only",
            "automatic_acceptance_allowed": False,
        },
        "policy": "shadow_only_no_keep_reject_delete_or_corpus_mutation",
        "artifact_sha256": {
            "source": sha256(args.source),
            "development_summary": sha256(args.development_summary),
            "excluded_uid_files": {str(path): sha256(path) for path in args.exclude_uids},
            "excluded_jsonl_files": {str(path): sha256(path) for path in args.exclude_jsonl},
            "sealed_selection": sha256(semantic_path),
            "blind_source_manifest": sha256(blind_path),
        },
    }
    (args.out / "preregistration.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(contract, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
