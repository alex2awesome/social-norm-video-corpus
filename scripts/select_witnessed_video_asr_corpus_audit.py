#!/usr/bin/env python3
"""Freeze a blind, source-disjoint corpus audit of witnessed video+ASR scores.

The uniform cohort supports corpus estimates.  Enrichment cohorts are kept
separate and exist only to expose false-positive and false-negative mechanisms
efficiently.  Selection never changes source media, metadata, or labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from scripts.witnessed_reaction_av_contract import candidate_positive
else:
    from witnessed_reaction_av_contract import candidate_positive


MANUAL_FIELDS = (
    "audit_candidate_index",
    "clip_audit_index",
    "candidate_id",
    "item_id",
    "uid",
    "reaction_grounded",
    "action_before_or_overlaps_response",
    "response_targets_action",
    "responder_role",
    "response_content",
    "trigger_kind",
    "staging",
    "manual_evidence",
    "manual_note",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_excluded_uids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            value = line.strip()
            if value:
                excluded.add(value)
    return excluded


def successful_scores(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return one successful result per candidate; successful retries win."""
    scores: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        if candidate_id not in scores or (
            scores[candidate_id].get("error") and not row.get("error")
        ):
            scores[candidate_id] = row
    return scores


def population_score_coverage(
    manifest_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    candidate_ids = [str(row.get("candidate_id") or "") for row in manifest_rows]
    if any(not value for value in candidate_ids):
        raise ValueError("manifest contains an empty candidate_id")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("manifest contains duplicate candidate_id values")
    expected = set(candidate_ids)
    scores = successful_scores(score_rows)
    successful = {
        candidate_id for candidate_id, row in scores.items()
        if candidate_id in expected
        and row.get("error") in (None, "")
        and isinstance(row.get("result"), dict)
    }
    unexpected = sorted(set(scores) - expected)
    return {
        "expected_candidates": len(expected),
        "successful_candidates": len(successful),
        "successful_coverage_fraction": len(successful) / len(expected),
        "unexpected_candidate_ids": unexpected[:20],
    }


def model_outcome(
    candidates: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
) -> str:
    model_rows = [scores.get(str(row["candidate_id"])) for row in candidates]
    if not model_rows or any(
        row is None or row.get("error") or not isinstance(row.get("result"), dict)
        for row in model_rows
    ):
        return "model_error_or_missing"
    results = [row["result"] for row in model_rows if row is not None]
    if any(
        candidate_positive(result, include_authority=False, require_social=False)
        for result in results
    ):
        return "predicted_strict_reaction"
    if any(
        result.get("reaction_grounded") in {"yes", "uncertain"}
        or result.get("response_targets_action") in {"yes", "uncertain"}
        or result.get("responder_role")
        in {"separate_bystander", "organic_audience", "offscreen_or_unresolved"}
        for result in results
    ):
        return "predicted_near_miss"
    return "predicted_clean_negative"


def representative_clips(
    manifest_rows: list[dict[str, Any]],
    excluded_uids: set[str],
    seed: str,
) -> list[dict[str, Any]]:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        if row.get("error") or not row.get("candidate_id") or not row.get("item_id"):
            continue
        by_item[str(row["item_id"])].append(row)
    by_uid: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for item_id, candidates in by_item.items():
        uid = str(candidates[0].get("uid") or "")
        if uid and uid not in excluded_uids:
            by_uid[uid].append((item_id, candidates))
    output = []
    for uid, clips in by_uid.items():
        # Define a reproducible one-clip-per-source audit population before
        # looking at model outputs, preserving source-disjointness.
        clips.sort(
            key=lambda pair: hashlib.sha256(
                f"{seed}:representative:{pair[0]}".encode()
            ).hexdigest()
        )
        item_id, candidates = clips[0]
        output.append({
            "item_id": item_id,
            "uid": uid,
            "platform": uid.split("__", 1)[0],
            "candidates": sorted(
                candidates, key=lambda row: str(row["candidate_id"])
            ),
        })
    return output


def balanced_pick(
    rows: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Round-robin platforms only for error-discovery enrichment cohorts."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["platform"])].append(row)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    names = sorted(groups)
    while len(selected) < count and names:
        next_names = []
        for name in names:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop())
            if groups[name]:
                next_names.append(name)
        names = next_names
    return selected


def select(
    manifest_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    excluded_uids: set[str],
    *,
    seed: str,
    uniform_clips: int,
    positive_enrichment: int,
    near_miss_enrichment: int,
    model_error_enrichment: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    scores = successful_scores(score_rows)
    population = representative_clips(manifest_rows, excluded_uids, seed)
    for row in population:
        row["model_outcome"] = model_outcome(row["candidates"], scores)
    population_counts = Counter(row["model_outcome"] for row in population)

    remaining = list(population)
    rng.shuffle(remaining)
    chosen: list[tuple[str, dict[str, Any]]] = []
    for row in remaining[:uniform_clips]:
        chosen.append(("uniform_probability_sample", row))
    used = {row["uid"] for _cohort, row in chosen}

    enrichment_specs = (
        ("predicted_positive_enrichment", "predicted_strict_reaction", positive_enrichment),
        ("predicted_near_miss_enrichment", "predicted_near_miss", near_miss_enrichment),
        ("model_error_enrichment", "model_error_or_missing", model_error_enrichment),
    )
    for cohort, outcome, requested in enrichment_specs:
        eligible = [
            row for row in population
            if row["uid"] not in used and row["model_outcome"] == outcome
        ]
        for row in balanced_pick(eligible, requested, rng):
            chosen.append((cohort, row))
            used.add(row["uid"])

    rng.shuffle(chosen)
    sealed = []
    for clip_index, (cohort, clip) in enumerate(chosen):
        score_payload = {
            str(candidate["candidate_id"]): scores.get(str(candidate["candidate_id"]))
            for candidate in clip["candidates"]
        }
        sealed.append({
            "clip_audit_index": clip_index,
            "item_id": clip["item_id"],
            "uid": clip["uid"],
            "platform": clip["platform"],
            "cohort": cohort,
            "model_outcome": clip["model_outcome"],
            "candidates": clip["candidates"],
            "model_scores": score_payload,
            "policy": "manual_audit_only_no_keep_reject_or_corpus_mutation",
        })
    metadata = {
        "eligible_source_uids": len(population),
        "eligible_representative_clips": len(population),
        "population_outcomes": dict(sorted(population_counts.items())),
        "selected_clips": len(sealed),
        "selected_candidates": sum(len(row["candidates"]) for row in sealed),
        "cohorts": dict(sorted(Counter(row["cohort"] for row in sealed).items())),
        "population_model_score_coverage": population_score_coverage(
            manifest_rows, score_rows
        ),
    }
    return sealed, metadata


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def write_manual_template(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_FIELDS, delimiter="\t")
        writer.writeheader()
        candidate_index = 0
        for clip in rows:
            for candidate in clip["candidates"]:
                writer.writerow({
                    "audit_candidate_index": candidate_index,
                    "clip_audit_index": clip["clip_audit_index"],
                    "candidate_id": candidate["candidate_id"],
                    "item_id": clip["item_id"],
                    "uid": clip["uid"],
                })
                candidate_index += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path, action="append", default=[])
    parser.add_argument("--seed", default="witnessed-video-asr-corpus-audit-v1")
    parser.add_argument("--uniform-clips", type=int, default=60)
    parser.add_argument("--positive-enrichment", type=int, default=30)
    parser.add_argument("--near-miss-enrichment", type=int, default=30)
    parser.add_argument("--model-error-enrichment", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.out_dir}")
    if min(
        args.uniform_clips,
        args.positive_enrichment,
        args.near_miss_enrichment,
        args.model_error_enrichment,
    ) < 0:
        raise SystemExit("sample counts must be non-negative")
    args.out_dir.mkdir(parents=True)
    excluded = read_excluded_uids(args.exclude_uids)
    sealed, metadata = select(
        read_jsonl(args.manifest),
        read_jsonl(args.scores),
        excluded,
        seed=args.seed,
        uniform_clips=args.uniform_clips,
        positive_enrichment=args.positive_enrichment,
        near_miss_enrichment=args.near_miss_enrichment,
        model_error_enrichment=args.model_error_enrichment,
    )
    sealed_path = args.out_dir / "sealed_selection.jsonl"
    write_jsonl(sealed_path, sealed)
    blind = []
    for clip in sealed:
        blind.append({
            "clip_audit_index": clip["clip_audit_index"],
            "item_id": clip["item_id"],
            "uid": clip["uid"],
            "candidates": [
                {
                    key: candidate.get(key)
                    for key in (
                        "candidate_id", "candidate_video_path",
                        "candidate_video_sha256", "candidate_relative_start_sec",
                        "candidate_relative_end_sec", "window_duration_sec",
                    )
                }
                for candidate in clip["candidates"]
            ],
        })
    blind_path = args.out_dir / "blind_visual_selection.jsonl"
    write_jsonl(blind_path, blind)
    template_path = args.out_dir / "manual_atomic_ledger.tsv"
    write_manual_template(template_path, sealed)
    prereg = {
        "kind": "witnessed_video_asr_corpus_intensive_audit_v1",
        "seed": args.seed,
        "manual_review_required": True,
        "source_disjoint_within_audit": len({row["uid"] for row in sealed}) == len(sealed),
        "excluded_prior_manual_uids": len(excluded),
        **metadata,
        "uniform_cohort_use": (
            "unbiased_for_the_deterministic_one_clip_per_source_population_"
            "not_clip_weighted_corpus_precision"
        ),
        "sampling_estimand": (
            "one_deterministically_selected_witnessed_item_per_previously_"
            "unaudited_source_uid"
        ),
        "enrichment_cohort_use": "error_discovery_only_report_separately",
        "blind_visual_policy": "model_outputs_cohorts_and_asr_hidden_in_first_pass",
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
        "sealed_selection_sha256": sha256(sealed_path),
        "blind_visual_selection_sha256": sha256(blind_path),
    }
    (args.out_dir / "preregistration.json").write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(prereg, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
