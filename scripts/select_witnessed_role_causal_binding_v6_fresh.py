#!/usr/bin/env python3
"""Select fresh witnessed V6 uniform and V3-positive cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from scripts.select_witnessed_video_asr_corpus_audit import (
        MANUAL_FIELDS, read_excluded_uids, read_jsonl, representative_clips,
        successful_scores, write_jsonl, write_manual_template,
    )
    from scripts.witnessed_reaction_av_contract import candidate_positive
else:
    from select_witnessed_video_asr_corpus_audit import (
        MANUAL_FIELDS, read_excluded_uids, read_jsonl, representative_clips,
        successful_scores, write_jsonl, write_manual_template,
    )
    from witnessed_reaction_av_contract import candidate_positive


def reaction_outcome(candidates: list[dict[str, Any]], scores: dict[str, dict[str, Any]]) -> str:
    rows = [scores.get(str(candidate["candidate_id"])) for candidate in candidates]
    if not rows or any(
        row is None or row.get("error") or not isinstance(row.get("result"), dict)
        for row in rows
    ):
        return "model_error_or_missing"
    if any(candidate_positive(
        row["result"], include_authority=False, require_social=False,
        require_unstaged=False,
    ) for row in rows):
        return "v3_staging_independent_reaction_positive"
    return "v3_reaction_negative"


def select(
    manifest_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]],
    excluded: set[str], *, uniform: int, enriched: int, seed: str,
) -> list[dict[str, Any]]:
    scores = successful_scores(score_rows)
    population = representative_clips(manifest_rows, excluded, seed)
    for row in population:
        row["v3_outcome"] = reaction_outcome(row["candidates"], scores)
    rng = random.Random(seed)
    rng.shuffle(population)
    uniform_rows = population[:uniform]
    used = {row["uid"] for row in uniform_rows}
    positives = [row for row in population if row["uid"] not in used and row["v3_outcome"] == "v3_staging_independent_reaction_positive"]
    rng.shuffle(positives)
    if len(positives) < enriched:
        raise ValueError("insufficient fresh V3 reaction-positive sources")
    chosen = [("uniform_probability_sample", row) for row in uniform_rows]
    chosen.extend(("v3_positive_enrichment", row) for row in positives[:enriched])
    rng.shuffle(chosen)
    sealed = []
    for index, (cohort, row) in enumerate(chosen):
        sealed.append({
            "clip_audit_index": index,
            "item_id": row["item_id"], "uid": row["uid"],
            "platform": row["platform"], "cohort": cohort,
            "v3_outcome": row["v3_outcome"], "candidates": row["candidates"],
            "policy": "manual_audit_only_no_keep_reject_or_corpus_mutation",
        })
    if len({row["uid"] for row in sealed}) != len(sealed):
        raise AssertionError("selection is not source-disjoint")
    return sealed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v3-scores", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--uniform", type=int, default=80)
    parser.add_argument("--positive-enrichment", type=int, default=120)
    parser.add_argument("--seed", default="witnessed-role-causal-binding-v6-fresh")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    excluded = read_excluded_uids(args.exclude_uids)
    sealed = select(
        read_jsonl(args.manifest), read_jsonl(args.v3_scores), excluded,
        uniform=args.uniform, enriched=args.positive_enrichment, seed=args.seed,
    )
    args.out_dir.mkdir(parents=True)
    write_jsonl(args.out_dir / "sealed_selection.jsonl", sealed)
    blind = [{
        "clip_audit_index": row["clip_audit_index"], "item_id": row["item_id"],
        "uid": row["uid"], "candidates": [{key: candidate.get(key) for key in (
            "candidate_id", "candidate_video_path", "candidate_video_sha256",
            "candidate_relative_start_sec", "candidate_relative_end_sec",
            "window_duration_sec",
        )} for candidate in row["candidates"]],
    } for row in sealed]
    write_jsonl(args.out_dir / "blind_visual_selection.jsonl", blind)
    write_manual_template(args.out_dir / "manual_atomic_ledger.tsv", sealed)
    summary = {
        "kind": "witnessed_role_causal_binding_v6_fresh_selection",
        "selected_clips": len(sealed),
        "selected_candidates": sum(len(row["candidates"]) for row in sealed),
        "source_disjoint": len({row["uid"] for row in sealed}) == len(sealed),
        "excluded_prior_uids": len(excluded),
        "cohorts": dict(sorted(Counter(row["cohort"] for row in sealed).items())),
        "reaction_target_ignores_staging": True,
        "human_gold_must_be_sealed_before_v6_scoring": True,
        "manual_review_every_v3_and_v6_output": True,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }
    (args.out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
