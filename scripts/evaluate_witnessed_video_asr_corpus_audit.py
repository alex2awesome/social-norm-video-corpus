#!/usr/bin/env python3
"""Evaluate the frozen witnessed video+ASR corpus audit.

Only the uniform cohort is reported as a corpus estimate. Enrichment cohorts
remain error-discovery evidence and are never pooled into a headline estimate.
All uncertain or missing judgments abstain for atomic metrics and fail closed
for composed routing metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_witnessed_reaction_atomic_candidates import metric, wilson
    from scripts.witnessed_reaction_av_contract import candidate_positive
    from scripts.run_witnessed_reaction_candidate_av_vlm import (
        CONTENTS,
        ROLES,
        STAGING,
        TRIGGERS,
        TRINARY,
    )
    from scripts.select_witnessed_video_asr_corpus_audit import successful_scores
else:
    from evaluate_witnessed_reaction_atomic_candidates import metric, wilson
    from witnessed_reaction_av_contract import candidate_positive
    from run_witnessed_reaction_candidate_av_vlm import (
        CONTENTS, ROLES, STAGING, TRIGGERS, TRINARY,
    )
    from select_witnessed_video_asr_corpus_audit import successful_scores


SOCIAL_TRIGGERS = {"interpersonal_treatment", "shared_public_conduct"}
BYSTANDER_ROLES = {"separate_bystander", "organic_audience"}
TARGETED_CONTENT = {
    "targeted_objection",
    "correction_or_sanction",
    "protective_intervention",
    "interposition_or_separation",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manual(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("manual ledger is empty")
    ids = [row.get("candidate_id", "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("manual ledger has empty or duplicate candidate_id")
    allowed = {
        "reaction_grounded": TRINARY,
        "action_before_or_overlaps_response": TRINARY,
        "response_targets_action": TRINARY,
        "responder_role": ROLES,
        "response_content": CONTENTS,
        "trigger_kind": TRIGGERS,
        "staging": STAGING,
    }
    for row in rows:
        for field, values in allowed.items():
            if row.get(field) not in values:
                raise ValueError(
                    f"{row['candidate_id']}: invalid or missing {field}: "
                    f"{row.get(field)!r}"
                )
        if not row.get("manual_evidence", "").strip():
            raise ValueError(f"{row['candidate_id']}: missing manual_evidence")
        if row["response_targets_action"] == "yes" and row["reaction_grounded"] != "yes":
            raise ValueError(
                f"{row['candidate_id']}: targeted response must be grounded"
            )
        if (
            row["response_content"] in TARGETED_CONTENT
            and row["response_targets_action"] != "yes"
        ):
            raise ValueError(
                f"{row['candidate_id']}: targeted content must target the action"
            )
        if row["responder_role"] == "none" and row["reaction_grounded"] == "yes":
            raise ValueError(
                f"{row['candidate_id']}: grounded response requires a responder role"
            )


def atomic_values(result: dict[str, Any]) -> dict[str, bool | None]:
    def trinary(field: str) -> bool | None:
        return {"yes": True, "no": False}.get(result.get(field))

    role = result.get("responder_role")
    content = result.get("response_content")
    trigger = result.get("trigger_kind")
    staging = result.get("staging")
    return {
        "reaction_grounded": trinary("reaction_grounded"),
        "action_before_or_overlaps_response": trinary(
            "action_before_or_overlaps_response"
        ),
        "response_targets_action": trinary("response_targets_action"),
        "bystander_identity": (
            None if role == "offscreen_or_unresolved" else role in BYSTANDER_ROLES
        ),
        "targeted_response_content": (
            None if content == "uncertain" else content in TARGETED_CONTENT
        ),
        "social_trigger": (
            None if trigger == "uncertain" else trigger in SOCIAL_TRIGGERS
        ),
        "clearly_staged": (
            None if staging == "uncertain" else staging == "clearly_staged"
        ),
    }


def composed_positive(result: dict[str, Any], *, require_social: bool) -> bool:
    return candidate_positive(
        result,
        include_authority=False,
        require_social=require_social,
    )


def composed_staged_reroute(
    result: dict[str, Any], *, require_social: bool
) -> bool:
    """Identify enacted staged scenes that fail only the organic-source gate."""
    if result.get("staging") != "clearly_staged":
        return False
    organic_view = dict(result, staging="no_clear_staging_evidence")
    return composed_positive(organic_view, require_social=require_social)


def evaluate(
    selection_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, str]],
    model_rows: list[dict[str, Any]],
    selection_preregistration: dict[str, Any],
) -> dict[str, Any]:
    validate_manual(manual_rows)
    selected_candidates: dict[str, dict[str, Any]] = {}
    candidate_to_clip: dict[str, str] = {}
    clip_to_cohort: dict[str, str] = {}
    clip_to_uid: dict[str, str] = {}
    for clip in selection_rows:
        item_id = str(clip["item_id"])
        if item_id in clip_to_cohort:
            raise ValueError("selection contains duplicate item_id")
        clip_to_cohort[item_id] = str(clip["cohort"])
        clip_to_uid[item_id] = str(clip["uid"])
        for candidate in clip.get("candidates") or []:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in selected_candidates:
                raise ValueError("selection contains duplicate candidate_id")
            selected_candidates[candidate_id] = candidate
            candidate_to_clip[candidate_id] = item_id
    if len(set(clip_to_uid.values())) != len(clip_to_uid):
        raise ValueError("selection is not source-disjoint by uid")
    manual = {row["candidate_id"]: row for row in manual_rows}
    if set(manual) != set(selected_candidates):
        raise ValueError("manual ledger does not exactly cover sealed selection")
    for candidate_id, row in manual.items():
        item_id = candidate_to_clip[candidate_id]
        if row.get("item_id") != item_id or row.get("uid") != clip_to_uid[item_id]:
            raise ValueError(f"{candidate_id}: manual item_id/uid lineage mismatch")
    model = successful_scores(model_rows)
    for candidate_id in manual:
        if candidate_id not in model or model[candidate_id].get("error"):
            continue
        item_id = candidate_to_clip[candidate_id]
        if (
            model[candidate_id].get("item_id") != item_id
            or model[candidate_id].get("uid") != clip_to_uid[item_id]
        ):
            raise ValueError(f"{candidate_id}: model item_id/uid lineage mismatch")

    manual_results = {candidate_id: dict(row) for candidate_id, row in manual.items()}
    model_results = {
        candidate_id: (
            model[candidate_id].get("result")
            if candidate_id in model and not model[candidate_id].get("error")
            else None
        )
        for candidate_id in manual
    }
    atom_names = tuple(atomic_values(next(iter(manual_results.values()))))
    atomic_reports = {}
    for atom in atom_names:
        gold = {
            candidate_id: bool(value)
            for candidate_id, result in manual_results.items()
            if (value := atomic_values(result)[atom]) is not None
        }
        predicted = {
            candidate_id: (
                atomic_values(result)[atom] if result is not None else None
            )
            for candidate_id, result in model_results.items()
            if candidate_id in gold
        }
        atomic_reports[atom] = metric(gold, predicted)

    manual_clip_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model_clip_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate_id, manual_result in manual_results.items():
        item_id = candidate_to_clip[candidate_id]
        manual_clip_candidates[item_id].append(manual_result)
        if model_results[candidate_id] is not None:
            model_clip_candidates[item_id].append(model_results[candidate_id])

    cohort_reports = {}
    for cohort in sorted(set(clip_to_cohort.values())):
        items = sorted(
            item for item, value in clip_to_cohort.items() if value == cohort
        )
        routes = {}
        for require_social, route in (
            (False, "strict_bystander_reaction"),
            (True, "strict_bystander_social"),
        ):
            gold = {
                item: any(
                    composed_positive(result, require_social=require_social)
                    for result in manual_clip_candidates[item]
                )
                for item in items
            }
            predicted = {
                item: any(
                    composed_positive(result, require_social=require_social)
                    for result in model_clip_candidates[item]
                )
                for item in items
            }
            route_report = metric(gold, predicted)
            gold_positives = sum(gold.values())
            route_report["gold_prevalence"] = (
                gold_positives / len(gold) if gold else None
            )
            route_report["gold_prevalence_wilson_95"] = wilson(
                gold_positives, len(gold)
            )
            route_report["manual_staged_instructional_reroute_clips"] = sum(
                any(
                    composed_staged_reroute(
                        result, require_social=require_social
                    )
                    for result in manual_clip_candidates[item]
                )
                for item in items
            )
            routes[route] = route_report
        cohort_reports[cohort] = {
            "clips": len(items),
            "routes": routes,
            "interpretation": (
                "unbiased_one_representative_clip_per_source_estimate_"
                "not_clip_weighted"
                if cohort == "uniform_probability_sample"
                else "enriched_error_discovery_not_a_corpus_estimate"
            ),
        }

    coverage = sum(result is not None for result in model_results.values())
    population_coverage = selection_preregistration.get(
        "population_model_score_coverage"
    ) or {}
    expected_population = int(population_coverage.get("expected_candidates") or 0)
    successful_population = int(population_coverage.get("successful_candidates") or 0)
    reported_fraction = population_coverage.get("successful_coverage_fraction")
    if (
        expected_population <= 0
        or successful_population < 0
        or successful_population > expected_population
        or reported_fraction is None
        or abs(float(reported_fraction) - successful_population / expected_population) > 1e-12
        or population_coverage.get("unexpected_candidate_ids")
    ):
        raise ValueError("invalid population model-coverage lineage")
    return {
        "kind": "witnessed_video_asr_corpus_intensive_audit_v1",
        "policy": "manual_audit_shadow_routing_only_no_automatic_acceptance",
        "selected_clips": len(clip_to_cohort),
        "selected_candidates": len(selected_candidates),
        "source_disjoint": len(set(clip_to_uid.values())) == len(clip_to_uid),
        "manual_candidate_review_complete": len(manual_results) == len(selected_candidates),
        "organic_witnessed_requires_no_clear_staging_evidence": True,
        "selected_sample_model_candidate_coverage": coverage / len(selected_candidates),
        "population_model_candidate_coverage": float(reported_fraction),
        "population_expected_candidates": expected_population,
        "population_successful_candidates": successful_population,
        "manual_uncertain_counts": dict(sorted(Counter(
            field
            for result in manual_results.values()
            for field, value in atomic_values(result).items()
            if value is None
        ).items())),
        "candidate_atomic_metrics_descriptive_enriched_mix": atomic_reports,
        "clip_cohort_metrics": cohort_reports,
        "headline_source_uniform_estimate_cohort": "uniform_probability_sample",
        "sampling_estimand": (
            "one_deterministically_selected_witnessed_item_per_previously_"
            "unaudited_source_uid"
        ),
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--selection-preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        read_jsonl(args.selection), read_tsv(args.manual), read_jsonl(args.model),
        json.loads(args.selection_preregistration.read_text()),
    )
    report["artifacts_sha256"] = {
        "selection": sha256(args.selection),
        "manual": sha256(args.manual),
        "model": sha256(args.model),
        "selection_preregistration": sha256(args.selection_preregistration),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
