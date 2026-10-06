#!/usr/bin/env python3
"""Evaluate exhaustive manual review of hierarchical commentary windows."""

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
except ModuleNotFoundError:
    from evaluate_witnessed_reaction_atomic_candidates import metric  # type: ignore[no-redef]


YES_NO = {"yes", "no"}
ALIGNMENT = {"exact", "partial", "mismatch", "no_visual"}
ROUTES = {"commentary_visual", "instructional_demo", "text_only", "relabel_required", "reject"}


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


def validate_manual(
    selected: dict[str, dict[str, Any]],
    blind_rows: list[dict[str, str]],
    post_rows: list[dict[str, str]],
    media_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    blind = indexed(blind_rows, "window_id", "blind manual review")
    post = indexed(post_rows, "window_id", "post-reveal manual review")
    media = indexed(media_rows, "window_id", "commentary review media")
    if set(blind) != set(selected) or set(post) != set(selected) or set(media) != set(selected):
        raise ValueError("manual reviews do not exactly cover selected windows")
    for window_id in selected:
        uid = str(selected[window_id]["uid"])
        if blind[window_id].get("uid") != uid or post[window_id].get("uid") != uid:
            raise ValueError(f"{window_id}: manual lineage mismatch")
        for field in (
            "performed_event_visible", "actor_target_grounded",
            "before_action_after_complete", "crucial_action_occluded_or_offframe",
            "label_bearing_text_absent",
        ):
            if blind[window_id].get(field) not in YES_NO:
                raise ValueError(f"{window_id}: missing blind {field}")
        if not blind[window_id].get("literal_action_description", "").strip():
            raise ValueError(f"{window_id}: missing literal action description")
        if not blind[window_id].get("manual_rationale", "").strip():
            raise ValueError(f"{window_id}: missing blind rationale")
        expected_audio_status = (
            "reviewed" if media[window_id].get("audio_present") is True
            else "source_has_no_audio"
        )
        if blind[window_id].get("source_audio_review_status") != expected_audio_status:
            raise ValueError(f"{window_id}: source audio was not completely reviewed")
        modalities = blind[window_id].get("event_evidence_modalities")
        if modalities not in {
            "visual_only", "audiovisual_speech_act", "audiovisual_other",
            "no_visual_event",
        }:
            raise ValueError(f"{window_id}: missing event evidence modalities")
        if (
            (blind[window_id]["performed_event_visible"] == "no")
            != (modalities == "no_visual_event")
        ):
            raise ValueError(f"{window_id}: event modalities contradict visual gold")
        if modalities.startswith("audiovisual_") and media[window_id].get("audio_present") is not True:
            raise ValueError(f"{window_id}: audiovisual event evidence lacks audio")
        for field in (
            "exact_named_action_visible", "start_boundary_clean", "end_boundary_clean",
            "vlm_output_visually_supported",
        ):
            if post[window_id].get(field) not in YES_NO:
                raise ValueError(f"{window_id}: missing post-reveal {field}")
        if post[window_id].get("label_alignment") not in ALIGNMENT:
            raise ValueError(f"{window_id}: invalid label alignment")
        if post[window_id].get("commentary_visual_route") not in ROUTES:
            raise ValueError(f"{window_id}: invalid commentary route")
        alignment = post[window_id]["label_alignment"]
        exact_named = post[window_id]["exact_named_action_visible"] == "yes"
        if exact_named != (alignment == "exact"):
            raise ValueError(f"{window_id}: named-action and alignment judgments contradict")
        corrected = post[window_id].get("corrected_behavior_label", "").strip()
        relabel = (
            alignment == "partial"
            or post[window_id]["commentary_visual_route"] == "relabel_required"
        )
        if relabel and not corrected:
            raise ValueError(f"{window_id}: relabel decision lacks corrected behavior")
        if not relabel and corrected:
            raise ValueError(f"{window_id}: corrected behavior without relabel decision")
        if (
            post[window_id]["commentary_visual_route"] == "commentary_visual"
            and alignment != "exact"
        ):
            raise ValueError(f"{window_id}: commentary_visual route requires exact label")
        if (
            post[window_id]["commentary_visual_route"] == "text_only"
            and blind[window_id]["performed_event_visible"] == "yes"
        ):
            raise ValueError(f"{window_id}: text_only route contradicts visible event")
        if not post[window_id].get("manual_rationale", "").strip():
            raise ValueError(f"{window_id}: missing post-reveal rationale")
    return blind, post


def strict_exact(blind: dict[str, str], post: dict[str, str]) -> bool:
    return (
        blind["performed_event_visible"] == "yes"
        and blind["actor_target_grounded"] == "yes"
        and blind["before_action_after_complete"] == "yes"
        and blind["crucial_action_occluded_or_offframe"] == "no"
        and blind["label_bearing_text_absent"] == "yes"
        and post["exact_named_action_visible"] == "yes"
        and post["label_alignment"] == "exact"
        and post["start_boundary_clean"] == "yes"
        and post["end_boundary_clean"] == "yes"
        and post["commentary_visual_route"] == "commentary_visual"
    )


def validate_output_review(
    rows: list[dict[str, str]], selected: dict[str, dict[str, Any]],
    model: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    review = indexed(rows, "item_id", "VLM output review")
    if set(review) != set(selected):
        raise ValueError("VLM output review does not exactly cover selected windows")
    audit_indices: set[int] = set()
    for item, row in review.items():
        try:
            audit_index = int(row.get("audit_index", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{item}: missing audit_index") from exc
        if audit_index in audit_indices:
            raise ValueError(f"{item}: duplicate audit_index")
        audit_indices.add(audit_index)
        if (
            row.get("uid") != str(selected[item]["uid"])
            or row.get("model") != str(model[item].get("model") or "")
        ):
            raise ValueError(f"{item}: output-review lineage mismatch")
        expected_prediction = "yes" if model[item].get("strict_pass") is True else "no"
        if row.get("model_strict_pass") != expected_prediction:
            raise ValueError(f"{item}: copied model prediction changed")
        for field in (
            "structured_prediction_supported", "rationale_supported",
            "input_modality_claim_supported",
            "stage_a_literal_analysis_supported", "stage_b_alignment_supported_by_stage_a",
        ):
            if row.get(field) not in YES_NO:
                raise ValueError(f"{item}: missing {field}")
        if not row.get("manual_rationale", "").strip():
            raise ValueError(f"{item}: missing output-review rationale")
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
        "stage_a_support_rate": sum(
            row["stage_a_literal_analysis_supported"] == "yes" for row in review.values()
        ) / len(review),
        "stage_b_support_rate": sum(
            row["stage_b_alignment_supported_by_stage_a"] == "yes" for row in review.values()
        ) / len(review),
    }


def evaluate(
    selected_rows: list[dict[str, Any]],
    blind_rows: list[dict[str, str]],
    post_rows: list[dict[str, str]],
    model_rows: list[dict[str, Any]],
    output_review_rows: list[dict[str, str]],
    media_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = indexed(selected_rows, "window_id", "selected windows")
    blind, post = validate_manual(selected, blind_rows, post_rows, media_rows)
    model = indexed(model_rows, "candidate_id", "model outputs")
    if set(model) != set(selected):
        raise ValueError("successful model outputs do not exactly cover selected windows")
    if any(row.get("error") not in (None, "") for row in model.values()):
        raise ValueError("model outputs contain errors")
    output_review = validate_output_review(output_review_rows, selected, model)

    gold = {item: strict_exact(blind[item], post[item]) for item in selected}
    predicted = {item: model[item].get("strict_pass") is True for item in selected}
    window_metrics = metric(gold, predicted)

    by_source: dict[str, list[str]] = defaultdict(list)
    for item, row in selected.items():
        by_source[str(row["uid"])].append(item)
    source_gold = {uid: any(gold[item] for item in items) for uid, items in by_source.items()}
    source_predicted = {
        uid: any(predicted[item] for item in items) for uid, items in by_source.items()
    }
    source_metrics = metric(source_gold, source_predicted)

    route_counts: dict[str, dict[str, Any]] = {}
    for route in sorted({reason for row in selected.values() for reason in row.get("selection_reasons") or []}):
        ids = [item for item, row in selected.items() if route in (row.get("selection_reasons") or [])]
        positives = sum(gold[item] for item in ids)
        route_counts[route] = {
            "selected_windows": len(ids),
            "strict_exact_windows": positives,
            "strict_exact_yield": positives / len(ids) if ids else None,
        }

    strict = window_metrics["fail_closed"]
    wilson = strict["precision_wilson_95"]
    gate_checks = {
        "manual_review_coverage": True,
        "model_output_coverage": True,
        "minimum_selected_15": strict["selected"] >= 15,
        "precision_at_least_0_90": (strict["precision"] or 0) >= 0.90,
        "precision_wilson_lower_at_least_0_80": bool(wilson) and wilson[0] >= 0.80,
        "rationale_support_at_least_0_90": output_review["rationale_support_rate"] >= 0.90,
    }
    return {
        "kind": "commentary_hierarchical_full_source_v2_evaluation",
        "sources": len(by_source),
        "selected_windows": len(selected),
        "manual_review_complete": True,
        "manual_source_audio_review_complete": True,
        "manual_gold_modalities": [
            "unmasked_ordered_video_frames", "selected_window_source_audio"
        ],
        "model_input_modalities": ["ocr_masked_ordered_video_frames"],
        "model_did_not_consume_source_audio": True,
        "strict_exact_window_metrics": window_metrics,
        "source_has_strict_exact_selected_window_metrics": source_metrics,
        "selection_route_yields_nonexclusive": route_counts,
        "manual_output_review": output_review,
        "promotion_gate_checks": gate_checks,
        "passes_this_development_gate": all(gate_checks.values()),
        "selected_windows_are_final_training_artifacts": False,
        "requires_muted_exact_artifact_materialization_and_reaudit": True,
        "individual_exact_clips_accepted": 0,
        "source_localization_recall_estimated": False,
        "source_localization_recall_reason": "unselected full-source windows lack exhaustive human gold",
        "requires_second_fresh_confirmation": True,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--post-reveal", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-review", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "selected": args.selected, "blind": args.blind,
        "post_reveal": args.post_reveal, "model": args.model,
        "output_review": args.output_review,
        "media_manifest": args.media_manifest,
    }
    report = evaluate(
        read_jsonl(args.selected), read_tsv(args.blind),
        read_tsv(args.post_reveal), read_jsonl(args.model),
        read_tsv(args.output_review), read_jsonl(args.media_manifest),
    )
    report["artifacts_sha256"] = {name: sha256(path) for name, path in paths.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
