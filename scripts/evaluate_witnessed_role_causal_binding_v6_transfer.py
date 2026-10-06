#!/usr/bin/env python3
"""Evaluate fresh witnessed V6 role/causal binding against V3 and manual gold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_witnessed_reaction_atomic_candidates import metric
    from scripts.evaluate_witnessed_video_asr_corpus_audit import atomic_values, validate_manual
    from scripts.freeze_fresh_manual_audit_phase import validate_witnessed
    from scripts.select_witnessed_video_asr_corpus_audit import successful_scores
    from scripts.witnessed_reaction_av_contract import candidate_positive
except ModuleNotFoundError:
    from evaluate_witnessed_reaction_atomic_candidates import metric  # type: ignore[no-redef]
    from evaluate_witnessed_video_asr_corpus_audit import atomic_values, validate_manual  # type: ignore[no-redef]
    from freeze_fresh_manual_audit_phase import validate_witnessed  # type: ignore[no-redef]
    from select_witnessed_video_asr_corpus_audit import successful_scores  # type: ignore[no-redef]
    from witnessed_reaction_av_contract import candidate_positive  # type: ignore[no-redef]


YES_NO = {"yes", "no"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reaction_positive(result: dict[str, Any]) -> bool:
    return candidate_positive(
        result,
        include_authority=False,
        require_social=False,
        require_unstaged=False,
    )


def validated_model(
    rows: list[dict[str, Any]], candidate_ids: set[str], name: str
) -> dict[str, dict[str, Any]]:
    scores = successful_scores(rows)
    if set(scores) != candidate_ids:
        raise ValueError(f"{name} successful outputs do not exactly cover candidates")
    output = {}
    for candidate_id, row in scores.items():
        if row.get("error") or not isinstance(row.get("result"), dict):
            raise ValueError(f"{name} has unsuccessful output: {candidate_id}")
        output[candidate_id] = row["result"]
    return output


def validate_output_reviews(
    rows: list[dict[str, str]], candidate_ids: set[str], model_name: str,
    score_rows: list[dict[str, Any]], candidate_uid: dict[str, str],
) -> dict[str, Any]:
    indexed = {row.get("item_id", ""): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != candidate_ids:
        raise ValueError(f"{model_name} output reviews do not exactly cover candidates")
    scores = successful_scores(score_rows)
    if set(scores) != candidate_ids:
        raise ValueError(f"{model_name} review lineage lacks exact successful scores")
    audit_indices: set[int] = set()
    for candidate_id, row in indexed.items():
        try:
            audit_index = int(row.get("audit_index", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{candidate_id}: missing audit_index") from exc
        if audit_index in audit_indices:
            raise ValueError(f"{candidate_id}: duplicate audit_index")
        audit_indices.add(audit_index)
        source = scores[candidate_id]
        if (
            row.get("uid") != candidate_uid[candidate_id]
            or row.get("model") != str(source.get("model") or "")
            or row.get("prompt_version") != str(source.get("prompt_version") or "")
        ):
            raise ValueError(f"{candidate_id}: output-review lineage mismatch")
        support_fields = (
            "structured_prediction_supported", "rationale_supported",
            "input_modality_claim_supported",
            "reaction_decision_supported", "role_binding_supported",
            "trigger_binding_supported", "staging_decision_supported",
        )
        for field in support_fields:
            if row.get(field) not in YES_NO:
                raise ValueError(f"{candidate_id}: missing {field} review")
        if not row.get("manual_rationale", "").strip():
            raise ValueError(f"{candidate_id}: missing output-review rationale")
        if (
            any(row[field] == "no" for field in support_fields)
            and not row.get("unsupported_claim_types", "").strip()
        ):
            raise ValueError(f"{candidate_id}: unsupported output lacks claim types")
    return {
        "reviewed": len(rows),
        "coverage": 1.0,
        "structured_prediction_support_rate": sum(
            row["structured_prediction_supported"] == "yes" for row in rows
        ) / len(rows),
        "rationale_support_rate": sum(
            row["rationale_supported"] == "yes" for row in rows
        ) / len(rows),
        "input_modality_claim_support_rate": sum(
            row["input_modality_claim_supported"] == "yes" for row in rows
        ) / len(rows),
        "reaction_decision_support_rate": sum(
            row["reaction_decision_supported"] == "yes" for row in rows
        ) / len(rows),
        "role_binding_support_rate": sum(
            row["role_binding_supported"] == "yes" for row in rows
        ) / len(rows),
        "trigger_binding_support_rate": sum(
            row["trigger_binding_supported"] == "yes" for row in rows
        ) / len(rows),
        "staging_decision_support_rate": sum(
            row["staging_decision_supported"] == "yes" for row in rows
        ) / len(rows),
    }


def atomic_report(
    manual: dict[str, dict[str, str]], model: dict[str, dict[str, Any]], ids: set[str]
) -> dict[str, Any]:
    output = {}
    for atom in (
        "reaction_grounded",
        "action_before_or_overlaps_response",
        "response_targets_action",
        "bystander_identity",
        "targeted_response_content",
        "social_trigger",
        "clearly_staged",
    ):
        gold_values = {item: atomic_values(manual[item])[atom] for item in ids}
        observed_gold = {item: value for item, value in gold_values.items() if value is not None}
        predictions = {item: atomic_values(model[item])[atom] for item in observed_gold}
        output[atom] = metric(observed_gold, predictions)
    return output


def evaluate(
    selection_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, str]],
    v3_rows: list[dict[str, Any]],
    v6_rows: list[dict[str, Any]],
    v3_review_rows: list[dict[str, str]],
    v6_review_rows: list[dict[str, str]],
    media_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_witnessed(selection_rows, manual_rows, media_rows)
    candidate_clip: dict[str, str] = {}
    candidate_uid: dict[str, str] = {}
    clip_cohort: dict[str, str] = {}
    clip_uid: dict[str, str] = {}
    for clip in selection_rows:
        item_id = str(clip.get("item_id") or "")
        uid = str(clip.get("uid") or "")
        cohort = str(clip.get("cohort") or "")
        if not item_id or not uid or not cohort or item_id in clip_cohort:
            raise ValueError("invalid or duplicate selected clip")
        clip_cohort[item_id] = cohort
        clip_uid[item_id] = uid
        for candidate in clip.get("candidates") or []:
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id or candidate_id in candidate_clip:
                raise ValueError("invalid or duplicate candidate")
            candidate_clip[candidate_id] = item_id
            candidate_uid[candidate_id] = uid
    if len(set(clip_uid.values())) != len(clip_uid):
        raise ValueError("selection is not source-disjoint")
    candidate_ids = set(candidate_clip)
    manual = {row["candidate_id"]: row for row in manual_rows}
    if set(manual) != candidate_ids:
        raise ValueError("manual ledger does not exactly cover candidates")
    for candidate_id, row in manual.items():
        clip = candidate_clip[candidate_id]
        if row.get("item_id") != clip or row.get("uid") != clip_uid[clip]:
            raise ValueError(f"{candidate_id}: manual lineage mismatch")

    v3 = validated_model(v3_rows, candidate_ids, "V3")
    v6 = validated_model(v6_rows, candidate_ids, "V6")
    review = {
        "v3": validate_output_reviews(
            v3_review_rows, candidate_ids, "V3", v3_rows, candidate_uid
        ),
        "v6": validate_output_reviews(
            v6_review_rows, candidate_ids, "V6", v6_rows, candidate_uid
        ),
    }

    candidates_by_clip: dict[str, list[str]] = defaultdict(list)
    for candidate_id, clip in candidate_clip.items():
        candidates_by_clip[clip].append(candidate_id)
    cohorts = {}
    for cohort in sorted(set(clip_cohort.values())):
        clips = {clip for clip, value in clip_cohort.items() if value == cohort}
        ids = {candidate for clip in clips for candidate in candidates_by_clip[clip]}
        gold = {
            clip: any(reaction_positive(manual[candidate]) for candidate in candidates_by_clip[clip])
            for clip in clips
        }
        predictions = {
            "v3": {
                clip: any(reaction_positive(v3[candidate]) for candidate in candidates_by_clip[clip])
                for clip in clips
            },
            "v6": {
                clip: any(reaction_positive(v6[candidate]) for candidate in candidates_by_clip[clip])
                for clip in clips
            },
        }
        cohorts[cohort] = {
            "clips": len(clips),
            "candidates": len(ids),
            "gold_positive_clips": sum(gold.values()),
            "clip_level": {
                model: metric(gold, values) for model, values in predictions.items()
            },
            "candidate_atomic": {
                "v3": atomic_report(manual, v3, ids),
                "v6": atomic_report(manual, v6, ids),
            },
            "interpretation": (
                "unbiased_one_clip_per_source_probability_sample"
                if cohort == "uniform_probability_sample"
                else "v3_positive_enrichment_not_population_prevalence"
            ),
        }

    enriched = cohorts.get("v3_positive_enrichment")
    gate_checks: dict[str, bool] = {
        "model_candidate_coverage": True,
        "all_output_reviews_complete": True,
        "v6_rationale_support_at_least_0_90": review["v6"]["rationale_support_rate"] >= 0.9,
        # The frozen V6 prompt described the MP4 audio track as model evidence,
        # but Qwen3-VL has no audio encoder.  Preserve the run for exhaustive
        # error analysis; never promote a modality-misstated transfer result.
        "v6_input_modality_matches_audiovisual_claim": False,
    }
    if enriched:
        v3_metric = enriched["clip_level"]["v3"]["fail_closed"]
        v6_metric = enriched["clip_level"]["v6"]["fail_closed"]
        wilson = v6_metric["precision_wilson_95"]
        gate_checks.update({
            "enriched_gold_positive_clips_at_least_25": enriched["gold_positive_clips"] >= 25,
            "v6_selected_clips_at_least_20": v6_metric["selected"] >= 20,
            "v6_precision_at_least_0_75": (v6_metric["precision"] or 0) >= 0.75,
            "v6_precision_wilson_lower_at_least_0_60": bool(wilson) and wilson[0] >= 0.60,
            "v6_recall_at_least_0_70": (v6_metric["recall"] or 0) >= 0.70,
            "v6_recall_drop_vs_v3_at_most_0_10": (
                (v6_metric["recall"] or 0) >= (v3_metric["recall"] or 0) - 0.10
            ),
        })
    else:
        gate_checks["required_enriched_cohort_present"] = False

    passes_this_transfer = all(gate_checks.values())
    return {
        "kind": "witnessed_role_causal_binding_v6_fresh_transfer_evaluation",
        "selected_clips": len(clip_cohort),
        "selected_candidates": len(candidate_ids),
        "source_disjoint": True,
        "manual_candidate_review_complete": True,
        "manual_source_audio_review_complete": True,
        "speaker_identity_basis_complete": True,
        "model_candidate_coverage": {"v3": 1.0, "v6": 1.0},
        "manual_output_review": review,
        "cohorts": cohorts,
        "promotion_gate_checks": gate_checks,
        "passes_this_transfer_gate": passes_this_transfer,
        "posthoc_modality_correction": {
            "model_consumed": ["video_frames", "supplied_asr_text"],
            "model_did_not_consume": ["mp4_audio_track"],
            "v6_allowed_use": "manual_error_discovery_only",
            "evidence_artifact": (
                "audit_runs/20260806_witnessed_v6_audio_capability_correction/"
                "summary.json"
            ),
        },
        "eligible_for_automatic_acceptance": False,
        "eligible_for_nondestructive_use_without_second_confirmation": False,
        "second_fresh_confirmation_required": True,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--v6", type=Path, required=True)
    parser.add_argument("--v3-output-review", type=Path, required=True)
    parser.add_argument("--v6-output-review", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "selection": args.selection,
        "manual": args.manual,
        "v3": args.v3,
        "v6": args.v6,
        "v3_output_review": args.v3_output_review,
        "v6_output_review": args.v6_output_review,
        "media_manifest": args.media_manifest,
    }
    report = evaluate(
        read_jsonl(args.selection), read_tsv(args.manual),
        read_jsonl(args.v3), read_jsonl(args.v6),
        read_tsv(args.v3_output_review), read_tsv(args.v6_output_review),
        read_jsonl(args.media_manifest),
    )
    report["artifacts_sha256"] = {name: sha256(path) for name, path in paths.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
