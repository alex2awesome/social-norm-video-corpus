#!/usr/bin/env python3
"""Select a source-disjoint V18 animation-rule audit and blind controls."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

try:
    from scripts.analyze_instructional_v18_shadow_rules import (
        feature_record,
        keyed_successes,
        read_jsonl,
    )
    from scripts.evaluate_instructional_v11_holdout import sha256
except ModuleNotFoundError:
    from analyze_instructional_v18_shadow_rules import (
        feature_record,
        keyed_successes,
        read_jsonl,
    )
    from evaluate_instructional_v11_holdout import sha256


CANDIDATE_SEED = "20260729-instructional-v18-animation-candidates-v1"
CONTROL_SEED = "20260729-instructional-v18-animation-controls-v1"
BLIND_SEED = "20260729-instructional-v18-animation-blind-v1"


def rank(seed: str, row: dict[str, Any]) -> str:
    payload = f"{seed}\0{row['item_id']}\0{row['uid']}".encode()
    return hashlib.sha256(payload).hexdigest()


def select(
    source_rows: list[dict[str, Any]],
    storyboard_rows: list[dict[str, Any]],
    glm_rows: dict[str, dict[str, Any]],
    qwen_rows: dict[str, dict[str, Any]],
    gemma_rows: dict[str, dict[str, Any]],
    excluded_uids: set[str],
    candidate_limit: int,
    control_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    storyboards = {
        str(row["item_id"]): row for row in storyboard_rows
    }
    population: list[dict[str, Any]] = []
    for source in source_rows:
        item_id = str(source["item_id"])
        if str(source["uid"]) in excluded_uids:
            continue
        if not all(
            item_id in records
            for records in (storyboards, glm_rows, qwen_rows, gemma_rows)
        ):
            raise ValueError(f"incomplete population record: {item_id}")
        source = {**source, "storyboard": storyboards[item_id]}
        features = feature_record(
            source, glm_rows[item_id], qwen_rows[item_id], gemma_rows[item_id]
        )
        if features["glm_demo"] and features["glm_animation"]:
            band = "candidate"
        elif (
            features["glm_demo"]
            and features["qwen_scene"]
            and not features["glm_animation"]
        ):
            band = "live_scene_control"
        elif features["glm_demo"] and not features["qwen_scene"]:
            band = "qwen_scene_reject_control"
        else:
            continue
        population.append(
            {
                **source,
                "band": band,
                "glm_v10a": glm_rows[item_id],
                "qwen_v16": qwen_rows[item_id],
                "gemma_causal": gemma_rows[item_id],
                "v18_visual_score": features["visual_score"],
            }
        )

    chosen: list[dict[str, Any]] = []
    for band in (
        "candidate",
        "live_scene_control",
        "qwen_scene_reject_control",
    ):
        rows = [row for row in population if row["band"] == band]
        limit = candidate_limit if band == "candidate" else control_limit
        seed = CANDIDATE_SEED if band == "candidate" else CONTROL_SEED + ":" + band
        chosen.extend(sorted(rows, key=lambda row: rank(seed, row))[:limit])

    random.Random(BLIND_SEED).shuffle(chosen)
    semantic: list[dict[str, Any]] = []
    blind: list[dict[str, Any]] = []
    for index, row in enumerate(chosen):
        candidate_id = f"v18holdout-{index:04d}"
        semantic.append(
            {
                **row,
                "audit_index": index,
                "candidate_id": candidate_id,
            }
        )
        board = row["storyboard"]
        blind.append(
            {
                "audit_index": index,
                "candidate_id": candidate_id,
                "sheet_path": board["sheet_path"],
                "sheet_sha256": board["sheet_sha256"],
            }
        )

    summary = {
        "kind": "instructional_v18_animation_holdout_selection",
        "status": "source_disjoint_preregistered_shadow_audit",
        "population_after_prior_audit_exclusion": len(
            {
                str(row["uid"])
                for row in source_rows
                if str(row["uid"]) not in excluded_uids
            }
        ),
        "excluded_prior_audit_uids": len(excluded_uids),
        "population_by_band": dict(
            sorted(Counter(row["band"] for row in population).items())
        ),
        "selected_by_band": dict(
            sorted(Counter(row["band"] for row in semantic).items())
        ),
        "selected_unique_uids": len({row["uid"] for row in semantic}),
        "candidate_seed": CANDIDATE_SEED,
        "control_seed": CONTROL_SEED,
        "blind_seed": BLIND_SEED,
        "candidate_limit": candidate_limit,
        "control_limit_per_band": control_limit,
        "semantic_fields_in_blind_manifest": False,
    }
    return semantic, blind, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--storyboards", type=Path, required=True)
    parser.add_argument("--glm", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument(
        "--exclude-semantic", type=Path, action="append", default=[]
    )
    parser.add_argument("--semantic-output", type=Path, required=True)
    parser.add_argument("--blind-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=30)
    parser.add_argument("--control-limit", type=int, default=15)
    args = parser.parse_args()
    excluded = {
        str(row["uid"])
        for path in args.exclude_semantic
        for row in read_jsonl(path)
    }
    semantic, blind, summary = select(
        read_jsonl(args.source),
        read_jsonl(args.storyboards),
        keyed_successes(args.glm),
        keyed_successes(args.qwen),
        keyed_successes(args.gemma),
        excluded,
        args.candidate_limit,
        args.control_limit,
    )
    args.semantic_output.parent.mkdir(parents=True, exist_ok=True)
    args.semantic_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in semantic)
    )
    args.blind_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind)
    )
    summary["artifact_sha256"] = {
        "semantic_output": sha256(args.semantic_output),
        "blind_output": sha256(args.blind_output),
    }
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
