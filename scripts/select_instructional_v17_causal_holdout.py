#!/usr/bin/env python3
"""Select the preregistered V17 causal-transfer holdout without label leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CANDIDATE_SEED = "20260729-instructional-v17-candidates-v1"
CONTROL_SEED = "20260729-instructional-v17-controls-v1"
BLIND_SEED = "20260729-instructional-v17-blind-v1"
QWEN_SCOPES = {
    "tacit_interpersonal",
    "shared_public",
    "explicit_social_etiquette",
}
QWEN_ROLES = {
    "situated_scene",
    "roleplay_demo",
    "animation_or_story_demo",
    "text_dialogue_demo",
}
GEMMA_CAUSAL = {"yes", "not_required"}
GEMMA_QUOTE_ROLES = {
    "performed_dialogue",
    "narrated_depicted_action",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item_id(row: dict[str, Any]) -> str:
    value = row.get("item_id")
    if not isinstance(value, str) or not value:
        raise ValueError("row has no non-empty item_id")
    return value


def unique_by_item(
    rows: Iterable[dict[str, Any]],
    *,
    name: str,
    require_result: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return the last successful record per item, rejecting unresolved items."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(item_id(row), []).append(row)
    selected: dict[str, dict[str, Any]] = {}
    for key, attempts in grouped.items():
        usable = [
            row
            for row in attempts
            if not row.get("error")
            and (not require_result or isinstance(row.get("result"), dict))
        ]
        if not usable:
            raise ValueError(f"{name}: no successful record for {key}")
        selected[key] = usable[-1]
    return selected


def rank(seed: str, band: str, key: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{band}\0{key}".encode("utf-8")
    ).hexdigest()


def qwen_passes(result: dict[str, Any]) -> bool:
    positive_intent = (
        result.get("demo_usable") == "yes"
        or result.get("demo_usable_raw") == "yes"
    )
    return (
        positive_intent
        and result.get("social_scope") in QWEN_SCOPES
        and result.get("scene_role") in QWEN_ROLES
        and result.get("rejection_reason") == "none"
    )


def gemma_passes(result: dict[str, Any]) -> bool:
    return (
        result.get("visual_record_supports_concrete_demo") == "yes"
        and result.get("same_actor_and_target") == "yes"
        and result.get("causal_or_intent_match") in GEMMA_CAUSAL
        and result.get("proposed_violation_polarity_matches") == "yes"
        and result.get("quote_role") in GEMMA_QUOTE_ROLES
    )


def select(
    population_rows: list[dict[str, Any]],
    storyboard_rows: list[dict[str, Any]],
    glm_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    gemma_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    *,
    candidate_limit: int = 40,
    control_limit: int = 15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    population = unique_by_item(population_rows, name="population")
    storyboards = unique_by_item(storyboard_rows, name="storyboards")
    glm = unique_by_item(glm_rows, name="glm", require_result=True)
    qwen = unique_by_item(qwen_rows, name="qwen", require_result=True)
    gemma = unique_by_item(gemma_rows, name="gemma", require_result=True)

    population_ids = set(population)
    for name, records in (
        ("storyboards", storyboards),
        ("glm", glm),
        ("qwen", qwen),
        ("gemma", gemma),
    ):
        missing = population_ids - set(records)
        extra = set(records) - population_ids
        if missing or extra:
            raise ValueError(
                f"{name} coverage mismatch: missing={len(missing)} "
                f"extra={len(extra)}"
            )

    excluded_uids = {
        row["uid"] for row in excluded_rows if isinstance(row.get("uid"), str)
    }
    eligible: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    for key, source in population.items():
        uid = source.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"population item {key} has no uid")
        if uid in excluded_uids:
            continue
        if uid in seen_uids:
            raise ValueError(f"eligible population repeats uid: {uid}")
        seen_uids.add(uid)
        if source.get("source_platform") != "dailymotion":
            raise ValueError(f"non-Dailymotion source in population: {key}")
        if source.get("polarity") != "violation":
            raise ValueError(f"non-violation source in population: {key}")

        glm_result = glm[key]["result"]
        qwen_result = qwen[key]["result"]
        gemma_result = gemma[key]["result"]
        if glm_result.get("demo_usable") != "yes":
            band = "glm_v10a_reject"
        elif not qwen_passes(qwen_result):
            band = "qwen_v16_reject"
        elif not gemma_passes(gemma_result):
            band = "gemma_causal_reject"
        else:
            band = "candidate"
        eligible.append(
            {
                **source,
                "band": band,
                "storyboard": storyboards[key],
                "glm_v10a": glm[key],
                "qwen_v16": qwen[key],
                "gemma_causal": gemma[key],
            }
        )

    by_band: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_band.setdefault(row["band"], []).append(row)

    chosen: list[dict[str, Any]] = []
    candidates = sorted(
        by_band.get("candidate", []),
        key=lambda row: rank(CANDIDATE_SEED, "candidate", item_id(row)),
    )
    chosen.extend(candidates[:candidate_limit])
    for band in (
        "glm_v10a_reject",
        "qwen_v16_reject",
        "gemma_causal_reject",
    ):
        controls = sorted(
            by_band.get(band, []),
            key=lambda row: rank(CONTROL_SEED, band, item_id(row)),
        )
        chosen.extend(controls[:control_limit])

    chosen.sort(key=lambda row: rank(BLIND_SEED, row["band"], item_id(row)))
    semantic: list[dict[str, Any]] = []
    blind: list[dict[str, Any]] = []
    for audit_index, row in enumerate(chosen):
        candidate_id = f"v17holdout-{audit_index:04d}"
        enriched = {
            **row,
            "audit_index": audit_index,
            "candidate_id": candidate_id,
        }
        semantic.append(enriched)
        board = row["storyboard"]
        blind.append(
            {
                "audit_index": audit_index,
                "candidate_id": candidate_id,
                "sheet_path": board["sheet_path"],
                "sheet_sha256": board["sheet_sha256"],
            }
        )

    band_population = Counter(row["band"] for row in eligible)
    band_selected = Counter(row["band"] for row in semantic)
    summary = {
        "kind": "instructional_v17_causal_transfer_selection",
        "population_input": len(population),
        "excluded_prior_uids": len(excluded_uids & {
            row.get("uid") for row in population.values()
        }),
        "eligible_source_distinct": len(eligible),
        "population_by_band": dict(sorted(band_population.items())),
        "selected": len(semantic),
        "selected_by_band": dict(sorted(band_selected.items())),
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
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--storyboards", type=Path, required=True)
    parser.add_argument("--glm", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--exclude-selection", type=Path, required=True)
    parser.add_argument("--semantic-out", type=Path, required=True)
    parser.add_argument("--blind-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument("--control-limit", type=int, default=15)
    args = parser.parse_args()
    for target in (args.semantic_out, args.blind_out, args.summary_out):
        if target.exists():
            raise SystemExit(f"refusing to overwrite: {target}")

    semantic, blind, summary = select(
        read_jsonl(args.population),
        read_jsonl(args.storyboards),
        read_jsonl(args.glm),
        read_jsonl(args.qwen),
        read_jsonl(args.gemma),
        read_jsonl(args.exclude_selection),
        candidate_limit=args.candidate_limit,
        control_limit=args.control_limit,
    )
    write_jsonl(args.semantic_out, semantic)
    write_jsonl(args.blind_out, blind)
    summary.update(
        {
            "semantic_out_sha256": digest(args.semantic_out),
            "blind_out_sha256": digest(args.blind_out),
        }
    )
    args.summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
