#!/usr/bin/env python3
"""Evaluate instructional temporal critic V4 on complete fresh manual gold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_witnessed_reaction_atomic_candidates import metric
except ModuleNotFoundError:
    from evaluate_witnessed_reaction_atomic_candidates import metric  # type: ignore[no-redef]


TRI = {"yes", "no", "uncertain"}
ALIGNMENT = {"exact", "partial", "mismatch", "no_visual"}
ERROR_TYPES = (
    "static_or_caption_assertion",
    "generic_conversation_or_activity",
    "presenter_or_interview",
    "disconnected_montage",
    "technical_or_nonsocial",
    "mixed_clip_segment_miss",
    "screen_action_miss",
    "offscreen_roleplay_miss",
    "nonhuman_public_coordination_miss",
    "unsupported_audio_or_modality_claim",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def indexed(rows: list[dict[str, Any]], field: str, name: str) -> dict[str, dict[str, Any]]:
    output = {str(row.get(field) or ""): row for row in rows}
    if not all(output) or len(output) != len(rows):
        raise ValueError(f"{name} has missing or duplicate {field}")
    return output


def validate_gold(
    cohort: dict[str, dict[str, Any]], rows: list[dict[str, str]],
    media_rows: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    gold = indexed(rows, "item_id", "manual gold")
    media = indexed(media_rows, "item_id", "instructional review media")
    if set(gold) != set(cohort):
        raise ValueError("manual gold does not exactly cover cohort")
    if set(media) != set(cohort):
        raise ValueError("instructional review media does not exactly cover cohort")
    for item_id, row in gold.items():
        if row.get("uid") != str(cohort[item_id]["uid"]):
            raise ValueError(f"{item_id}: manual lineage mismatch")
        if row.get("visual_demo") not in {"yes", "no"}:
            raise ValueError(f"{item_id}: visual_demo must be yes/no")
        for field in (
            "temporal_state_change", "recipient_response_or_coordinated_trajectory"
        ):
            if row.get(field) not in TRI:
                raise ValueError(f"{item_id}: missing {field}")
        if row.get("label_alignment") not in ALIGNMENT:
            raise ValueError(f"{item_id}: invalid label_alignment")
        if (
            (row["visual_demo"] == "no")
            != (row["label_alignment"] == "no_visual")
        ):
            raise ValueError(f"{item_id}: visual_demo and label_alignment contradict")
        corrected = row.get("corrected_behavior_label", "").strip()
        if row["label_alignment"] == "partial" and not corrected:
            raise ValueError(f"{item_id}: partial alignment lacks corrected label")
        if row["label_alignment"] != "partial" and corrected:
            raise ValueError(f"{item_id}: corrected label without partial alignment")
        try:
            start = int(row.get("segment_start_frame", ""))
            end = int(row.get("segment_end_frame", ""))
        except ValueError as exc:
            raise ValueError(f"{item_id}: invalid segment bounds") from exc
        if row["visual_demo"] == "yes":
            if not 0 <= start <= end <= 35:
                raise ValueError(f"{item_id}: positive demo lacks valid bounds")
        elif (start, end) != (-1, -1):
            raise ValueError(f"{item_id}: negative demo must use -1 bounds")
        if not row.get("manual_description", "").strip():
            raise ValueError(f"{item_id}: manual description is required")
        expected_audio_status = (
            "reviewed" if media[item_id].get("audio_present") is True
            else "source_has_no_audio"
        )
        if row.get("source_audio_review_status") != expected_audio_status:
            raise ValueError(f"{item_id}: source audio was not completely reviewed")
        modalities = row.get("demo_evidence_modalities")
        allowed_modalities = {
            "visual_only", "audiovisual_speech_act", "audiovisual_other", "no_demo"
        }
        if modalities not in allowed_modalities:
            raise ValueError(f"{item_id}: missing demo evidence modalities")
        if (row["visual_demo"] == "no") != (modalities == "no_demo"):
            raise ValueError(f"{item_id}: demo modalities contradict gold")
        if modalities.startswith("audiovisual_") and media[item_id].get("audio_present") is not True:
            raise ValueError(f"{item_id}: audiovisual gold lacks source audio")
    return gold


def validate_output_review(
    rows: list[dict[str, str]], cohort: dict[str, dict[str, Any]],
    model: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    review = indexed(rows, "item_id", "model output review")
    if set(review) != set(cohort):
        raise ValueError("model output review does not exactly cover cohort")
    audit_indices: set[int] = set()
    errors = Counter()
    for item_id, row in review.items():
        try:
            audit_index = int(row.get("audit_index", ""))
        except ValueError as exc:
            raise ValueError(f"{item_id}: missing audit_index") from exc
        if audit_index in audit_indices:
            raise ValueError(f"{item_id}: duplicate audit_index")
        audit_indices.add(audit_index)
        if (
            row.get("uid") != str(cohort[item_id]["uid"])
            or row.get("model") != str(model[item_id].get("model") or "")
        ):
            raise ValueError(f"{item_id}: output-review lineage mismatch")
        expected_prediction = (
            "yes" if model[item_id]["result"].get("demo_candidate") is True else "no"
        )
        if row.get("model_demo_candidate") != expected_prediction:
            raise ValueError(f"{item_id}: copied model prediction changed")
        for field in (
            "structured_prediction_supported", "rationale_supported",
            "input_modality_claim_supported", "predicted_segment_bounds_supported",
        ):
            if row.get(field) not in {"yes", "no"}:
                raise ValueError(f"{item_id}: missing {field}")
        if not row.get("manual_rationale", "").strip():
            raise ValueError(f"{item_id}: missing output-review rationale")
        for value in filter(None, row.get("unsupported_claim_types", "").split(",")):
            value = value.strip()
            if value not in ERROR_TYPES:
                raise ValueError(f"{item_id}: unknown error type {value}")
            errors[value] += 1
    return {
        "reviewed": len(review),
        "coverage": 1.0,
        "structured_prediction_support_rate": sum(
            row["structured_prediction_supported"] == "yes" for row in review.values()
        ) / len(review),
        "rationale_support_rate": sum(
            row["rationale_supported"] == "yes" for row in review.values()
        ) / len(review),
        "input_modality_claim_support_rate": sum(
            row["input_modality_claim_supported"] == "yes" for row in review.values()
        ) / len(review),
        "segment_bounds_support_rate": sum(
            row["predicted_segment_bounds_supported"] == "yes" for row in review.values()
        ) / len(review),
        "unsupported_claim_type_counts": dict(sorted(errors.items())),
        "required_error_types_all_reported": all(value in errors for value in ERROR_TYPES),
    }


def evaluate(
    cohort_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, str]],
    model_rows: list[dict[str, Any]],
    output_review_rows: list[dict[str, str]],
    media_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cohort = indexed(cohort_rows, "item_id", "cohort")
    if len({row.get("uid") for row in cohort.values()}) != len(cohort):
        raise ValueError("cohort is not source-disjoint")
    gold_rows = validate_gold(cohort, manual_rows, media_rows)
    model = indexed(model_rows, "item_id", "model outputs")
    if set(model) != set(cohort):
        raise ValueError("model outputs do not exactly cover cohort")
    if any(row.get("error") not in (None, "") or not isinstance(row.get("result"), dict)
           for row in model.values()):
        raise ValueError("model outputs contain errors")
    output_review = validate_output_review(output_review_rows, cohort, model)
    gold = {item: row["visual_demo"] == "yes" for item, row in gold_rows.items()}
    predicted = {
        item: model[item]["result"].get("demo_candidate") is True for item in cohort
    }
    overall = metric(gold, predicted)

    strata: dict[str, Any] = {}
    for field in ("polarity", "category", "source_platform"):
        for value in sorted({str(row.get(field) or "<missing>") for row in cohort.values()}):
            ids = {item for item, row in cohort.items() if str(row.get(field) or "<missing>") == value}
            strata[f"{field}={value}"] = metric(
                {item: gold[item] for item in ids},
                {item: predicted[item] for item in ids},
            )
    duration_bins = {
        "under_2": lambda seconds: seconds < 2,
        "2_to_under_5": lambda seconds: 2 <= seconds < 5,
        "5_to_under_15": lambda seconds: 5 <= seconds < 15,
        "15_to_under_45": lambda seconds: 15 <= seconds < 45,
        "45_or_more": lambda seconds: seconds >= 45,
    }
    for name, predicate in duration_bins.items():
        ids = {
            item for item, row in cohort.items()
            if predicate(float(row["end_sec"]) - float(row["start_sec"]))
        }
        strata[f"duration={name}"] = metric(
            {item: gold[item] for item in ids}, {item: predicted[item] for item in ids}
        )

    strict = overall["fail_closed"]
    wilson = strict["precision_wilson_95"]
    gate_checks = {
        "minimum_successful_output_coverage_1_0": True,
        "manual_gold_coverage_1_0": True,
        "manual_output_review_coverage_1_0": True,
        "minimum_selected_15": strict["selected"] >= 15,
        "precision_at_least_0_90": (strict["precision"] or 0) >= 0.90,
        "precision_wilson_lower_at_least_0_75": bool(wilson) and wilson[0] >= 0.75,
        "recall_at_least_0_50": (strict["recall"] or 0) >= 0.50,
    }
    return {
        "kind": "instructional_temporal_critic_v4_fresh_transfer_evaluation",
        "items": len(cohort),
        "source_disjoint": True,
        "manual_gold_complete": True,
        "manual_source_audio_review_complete": True,
        "manual_gold_modalities": ["ordered_video_frames", "source_audio"],
        "model_input_modalities": ["ordered_video_frames", "interval_asr_quotes"],
        "model_did_not_consume_source_audio": True,
        "model_output_coverage": 1.0,
        "overall": overall,
        "strata": strata,
        "manual_output_review": output_review,
        "promotion_gate_checks": gate_checks,
        "passes_this_transfer_gate": all(gate_checks.values()),
        "activation_after_this_run": False,
        "localized_model_segments_are_final_training_artifacts": False,
        "requires_exact_segment_recut_and_label_leakage_reaudit": True,
        "individual_exact_clips_accepted": 0,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-review", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "cohort": args.cohort, "manual": args.manual,
        "model": args.model, "output_review": args.output_review,
        "media_manifest": args.media_manifest,
    }
    report = evaluate(
        read_jsonl(args.cohort), read_tsv(args.manual),
        read_jsonl(args.model), read_tsv(args.output_review),
        read_jsonl(args.media_manifest),
    )
    report["artifacts_sha256"] = {name: sha256(path) for name, path in paths.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
