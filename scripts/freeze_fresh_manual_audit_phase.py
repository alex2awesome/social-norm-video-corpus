#!/usr/bin/env python3
"""Validate and freeze each blind/manual phase before revealing later evidence.

The frozen TSV is an append-only audit artifact.  Later phases must match all
earlier human judgments exactly, which prevents labels or model outputs from
silently changing the blind gold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_witnessed_video_asr_corpus_audit import (
        validate_manual as validate_witnessed_manual,
    )
except ModuleNotFoundError:
    from evaluate_witnessed_video_asr_corpus_audit import (  # type: ignore[no-redef]
        validate_manual as validate_witnessed_manual,
    )


YES_NO = {"yes", "no"}
TRI = {"yes", "no", "uncertain"}
ALIGNMENT = {"exact", "partial", "mismatch", "no_visual"}
COMMENTARY_ROUTES = {
    "commentary_visual", "instructional_demo", "text_only",
    "relabel_required", "reject",
}
WITNESSED_AUDIO_REVIEW = {"reviewed", "source_has_no_audio"}
INSTRUCTIONAL_DEMO_MODALITIES = {
    "visual_only", "audiovisual_speech_act", "audiovisual_other", "no_demo",
}
COMMENTARY_EVENT_MODALITIES = {
    "visual_only", "audiovisual_speech_act", "audiovisual_other",
    "no_visual_event",
}
SPEAKER_IDENTITY_BASES = {
    "visible_speaking_continuity", "voice_and_visible_turn_binding",
    "multi_camera_continuity", "offscreen_or_unresolved",
    "not_applicable_no_reaction",
}
POSITIVE_SPEAKER_BASES = {
    "visible_speaking_continuity", "voice_and_visible_turn_binding",
    "multi_camera_continuity",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def indexed(rows: list[dict[str, Any]], field: str, name: str) -> dict[str, dict[str, Any]]:
    output = {str(row.get(field) or ""): row for row in rows}
    if not output or not all(output) or len(output) != len(rows):
        raise ValueError(f"{name} has missing or duplicate {field}")
    return output


def require_lineage(
    cohort: dict[str, dict[str, Any]], ledger: dict[str, dict[str, Any]], key: str,
) -> None:
    if set(cohort) != set(ledger):
        raise ValueError("manual ledger does not exactly cover the sealed cohort")
    for item_id, row in ledger.items():
        expected_uid = str(cohort[item_id].get("uid") or "")
        if expected_uid and row.get("uid") != expected_uid:
            raise ValueError(f"{item_id}: uid lineage mismatch")


def validate_instruction_blind(
    cohort_rows: list[dict[str, Any]], rows: list[dict[str, str]],
    media_rows: list[dict[str, Any]],
) -> None:
    cohort = indexed(cohort_rows, "item_id", "instructional cohort")
    ledger = indexed(rows, "item_id", "instructional blind ledger")
    media = indexed(media_rows, "item_id", "instructional review media")
    require_lineage(cohort, ledger, "item_id")
    if set(media) != set(cohort):
        raise ValueError("instructional review media does not exactly cover cohort")
    for item_id, row in ledger.items():
        if row.get("visual_demo") not in YES_NO:
            raise ValueError(f"{item_id}: visual_demo must be yes/no")
        for field in (
            "temporal_state_change", "recipient_response_or_coordinated_trajectory",
        ):
            if row.get(field) not in TRI:
                raise ValueError(f"{item_id}: missing {field}")
        try:
            start, end = int(row.get("segment_start_frame", "")), int(
                row.get("segment_end_frame", "")
            )
        except ValueError as exc:
            raise ValueError(f"{item_id}: invalid segment bounds") from exc
        if row["visual_demo"] == "yes" and not 0 <= start <= end <= 35:
            raise ValueError(f"{item_id}: positive demo lacks valid bounds")
        if row["visual_demo"] == "no" and (start, end) != (-1, -1):
            raise ValueError(f"{item_id}: negative demo must use -1 bounds")
        if row.get("label_alignment", "").strip():
            raise ValueError(f"{item_id}: label_alignment was revealed during blind phase")
        if row.get("corrected_behavior_label", "").strip():
            raise ValueError(
                f"{item_id}: corrected_behavior_label was revealed during blind phase"
            )
        if not row.get("manual_description", "").strip():
            raise ValueError(f"{item_id}: manual_description is required")
        expected_audio_status = (
            "reviewed" if media[item_id].get("audio_present") is True
            else "source_has_no_audio"
        )
        if row.get("source_audio_review_status") != expected_audio_status:
            raise ValueError(f"{item_id}: source audio was not completely reviewed")
        modalities = row.get("demo_evidence_modalities")
        if modalities not in INSTRUCTIONAL_DEMO_MODALITIES:
            raise ValueError(f"{item_id}: missing demo_evidence_modalities")
        if (row["visual_demo"] == "no") != (modalities == "no_demo"):
            raise ValueError(f"{item_id}: demo modality contradicts visual_demo")
        if (
            modalities.startswith("audiovisual_")
            and media[item_id].get("audio_present") is not True
        ):
            raise ValueError(f"{item_id}: audiovisual evidence lacks source audio")


def require_unchanged(
    prior_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
    key: str,
    locked_fields: tuple[str, ...],
) -> None:
    prior = indexed(prior_rows, key, "prior frozen ledger")
    current = indexed(current_rows, key, "current ledger")
    if set(prior) != set(current):
        raise ValueError("current ledger differs from prior frozen cohort")
    for item_id in prior:
        for field in locked_fields:
            if prior[item_id].get(field) != current[item_id].get(field):
                raise ValueError(f"{item_id}: frozen field changed: {field}")


def validate_instruction_label(
    cohort_rows: list[dict[str, Any]],
    rows: list[dict[str, str]],
    prior_rows: list[dict[str, str]],
    media_rows: list[dict[str, Any]],
) -> None:
    validate_instruction_blind(
        cohort_rows,
        [
            {
                **row,
                "label_alignment": "",
                "corrected_behavior_label": "",
            }
            for row in rows
        ],
        media_rows,
    )
    require_unchanged(
        prior_rows,
        rows,
        "item_id",
        (
            "uid", "visual_demo", "temporal_state_change",
            "recipient_response_or_coordinated_trajectory",
            "segment_start_frame", "segment_end_frame", "manual_description",
            "manual_note", "source_audio_review_status", "demo_evidence_modalities",
        ),
    )
    for row in rows:
        if row.get("label_alignment") not in ALIGNMENT:
            raise ValueError(f"{row['item_id']}: invalid label_alignment")
        if (
            (row["visual_demo"] == "no")
            != (row["label_alignment"] == "no_visual")
        ):
            raise ValueError(
                f"{row['item_id']}: visual_demo and label_alignment contradict"
            )
        if (
            row["label_alignment"] == "partial"
            and not row.get("corrected_behavior_label", "").strip()
        ):
            raise ValueError(
                f"{row['item_id']}: partial alignment requires corrected_behavior_label"
            )
        if (
            row["label_alignment"] != "partial"
            and row.get("corrected_behavior_label", "").strip()
        ):
            raise ValueError(
                f"{row['item_id']}: corrected label is only valid for partial alignment"
            )


def flatten_witnessed_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for clip in rows:
        for candidate in clip.get("candidates") or []:
            output.append({
                "candidate_id": candidate["candidate_id"],
                "item_id": clip["item_id"],
                "uid": clip["uid"],
            })
    indexed(output, "candidate_id", "witnessed selection")
    return output


def validate_witnessed(
    cohort_rows: list[dict[str, Any]], rows: list[dict[str, str]],
    media_rows: list[dict[str, Any]],
) -> None:
    validate_witnessed_manual(rows)
    cohort = indexed(
        flatten_witnessed_selection(cohort_rows), "candidate_id", "witnessed cohort"
    )
    ledger = indexed(rows, "candidate_id", "witnessed blind ledger")
    media = indexed(media_rows, "candidate_id", "witnessed manual media")
    require_lineage(cohort, ledger, "candidate_id")
    if set(media) != set(cohort):
        raise ValueError("witnessed manual media does not exactly cover cohort")
    for candidate_id, row in ledger.items():
        if row.get("item_id") != str(cohort[candidate_id]["item_id"]):
            raise ValueError(f"{candidate_id}: clip lineage mismatch")
        expected_audio_status = (
            "reviewed" if media[candidate_id].get("audio_present") is True
            else "source_has_no_audio"
        )
        if row.get("source_audio_review_status") != expected_audio_status:
            raise ValueError(f"{candidate_id}: source audio was not completely reviewed")
        basis = row.get("speaker_identity_basis")
        if basis not in SPEAKER_IDENTITY_BASES:
            raise ValueError(f"{candidate_id}: missing speaker_identity_basis")
        if (
            row.get("responder_role") in {"separate_bystander", "organic_audience"}
            and basis not in POSITIVE_SPEAKER_BASES
        ):
            raise ValueError(
                f"{candidate_id}: bystander identity lacks positive speaker binding"
            )
        if (
            row.get("responder_role") == "offscreen_or_unresolved"
            and basis != "offscreen_or_unresolved"
        ):
            raise ValueError(f"{candidate_id}: unresolved responder basis contradicts role")


def validate_commentary_blind(
    cohort_rows: list[dict[str, Any]], rows: list[dict[str, str]],
    media_rows: list[dict[str, Any]],
) -> None:
    cohort = indexed(cohort_rows, "window_id", "commentary cohort")
    ledger = indexed(rows, "window_id", "commentary blind ledger")
    media = indexed(media_rows, "window_id", "commentary review media")
    require_lineage(cohort, ledger, "window_id")
    if set(media) != set(cohort):
        raise ValueError("commentary review media does not exactly cover cohort")
    for window_id, row in ledger.items():
        for field in (
            "performed_event_visible", "actor_target_grounded",
            "before_action_after_complete", "crucial_action_occluded_or_offframe",
            "label_bearing_text_absent",
        ):
            if row.get(field) not in YES_NO:
                raise ValueError(f"{window_id}: missing blind field {field}")
        for field in ("literal_action_description", "manual_rationale"):
            if not row.get(field, "").strip():
                raise ValueError(f"{window_id}: missing {field}")
        expected_audio_status = (
            "reviewed" if media[window_id].get("audio_present") is True
            else "source_has_no_audio"
        )
        if row.get("source_audio_review_status") != expected_audio_status:
            raise ValueError(f"{window_id}: source audio was not completely reviewed")
        modalities = row.get("event_evidence_modalities")
        if modalities not in COMMENTARY_EVENT_MODALITIES:
            raise ValueError(f"{window_id}: missing event_evidence_modalities")
        if (
            (row["performed_event_visible"] == "no")
            != (modalities == "no_visual_event")
        ):
            raise ValueError(f"{window_id}: event modalities contradict visual judgment")
        if modalities.startswith("audiovisual_") and media[window_id].get("audio_present") is not True:
            raise ValueError(f"{window_id}: audiovisual event evidence lacks audio")


def validate_commentary_label(
    cohort_rows: list[dict[str, Any]], rows: list[dict[str, str]],
) -> None:
    cohort = indexed(cohort_rows, "window_id", "commentary cohort")
    ledger = indexed(rows, "window_id", "commentary label ledger")
    require_lineage(cohort, ledger, "window_id")
    for window_id, row in ledger.items():
        for field in (
            "exact_named_action_visible", "start_boundary_clean", "end_boundary_clean",
        ):
            if row.get(field) not in YES_NO:
                raise ValueError(f"{window_id}: missing label-phase field {field}")
        if row.get("label_alignment") not in ALIGNMENT:
            raise ValueError(f"{window_id}: invalid label_alignment")
        if row.get("commentary_visual_route") not in COMMENTARY_ROUTES:
            raise ValueError(f"{window_id}: invalid commentary_visual_route")
        if (
            row.get("label_alignment") == "partial"
            or row.get("commentary_visual_route") == "relabel_required"
        ) and not row.get("corrected_behavior_label", "").strip():
            raise ValueError(
                f"{window_id}: relabeling requires corrected_behavior_label"
            )
        if (
            row.get("label_alignment") != "partial"
            and row.get("commentary_visual_route") != "relabel_required"
            and row.get("corrected_behavior_label", "").strip()
        ):
            raise ValueError(
                f"{window_id}: corrected label is only valid for an explicit relabel"
            )
        if row.get("vlm_output_visually_supported", "").strip():
            raise ValueError(f"{window_id}: VLM support was revealed during label phase")
        if not row.get("manual_rationale", "").strip():
            raise ValueError(f"{window_id}: missing manual_rationale")


def validate_commentary_model_reveal(
    cohort_rows: list[dict[str, Any]],
    rows: list[dict[str, str]],
    prior_rows: list[dict[str, str]],
) -> None:
    require_unchanged(
        prior_rows,
        rows,
        "window_id",
        (
            "uid", "exact_named_action_visible", "label_alignment",
            "start_boundary_clean", "end_boundary_clean",
            "commentary_visual_route", "manual_rationale",
            "corrected_behavior_label",
        ),
    )
    cohort = indexed(cohort_rows, "window_id", "commentary cohort")
    ledger = indexed(rows, "window_id", "commentary model-reveal ledger")
    require_lineage(cohort, ledger, "window_id")
    for window_id, row in ledger.items():
        if row.get("vlm_output_visually_supported") not in YES_NO:
            raise ValueError(f"{window_id}: missing VLM output support judgment")


def freeze(
    phase: str,
    cohort_path: Path,
    ledger_path: Path,
    out: Path,
    prior_freeze: Path | None = None,
    media_manifest: Path | None = None,
) -> dict[str, Any]:
    if out.exists() or out.with_suffix(out.suffix + ".freeze.json").exists():
        raise FileExistsError(out)
    cohort_rows, rows = read_jsonl(cohort_path), read_tsv(ledger_path)
    prior_rows = read_tsv(prior_freeze) if prior_freeze else None
    if phase == "instructional_blind":
        if media_manifest is None:
            raise ValueError("instructional_blind requires --media-manifest")
        validate_instruction_blind(cohort_rows, rows, read_jsonl(media_manifest))
    elif phase == "instructional_label":
        if prior_rows is None:
            raise ValueError("instructional_label requires --prior-freeze")
        if media_manifest is None:
            raise ValueError("instructional_label requires --media-manifest")
        validate_instruction_label(
            cohort_rows, rows, prior_rows, read_jsonl(media_manifest)
        )
    elif phase == "witnessed_blind":
        if media_manifest is None:
            raise ValueError("witnessed_blind requires --media-manifest")
        validate_witnessed(cohort_rows, rows, read_jsonl(media_manifest))
    elif phase == "commentary_blind":
        if media_manifest is None:
            raise ValueError("commentary_blind requires --media-manifest")
        validate_commentary_blind(cohort_rows, rows, read_jsonl(media_manifest))
    elif phase == "commentary_label":
        validate_commentary_label(cohort_rows, rows)
    elif phase == "commentary_model_reveal":
        if prior_rows is None:
            raise ValueError("commentary_model_reveal requires --prior-freeze")
        validate_commentary_model_reveal(cohort_rows, rows, prior_rows)
    else:
        raise ValueError(f"unknown phase: {phase}")

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ledger_path, out)
    summary = {
        "kind": "fresh_manual_audit_phase_freeze",
        "phase": phase,
        "rows": len(rows),
        "ledger_sha256": sha256(out),
        "cohort_sha256": sha256(cohort_path),
        "prior_freeze_sha256": sha256(prior_freeze) if prior_freeze else None,
        "media_manifest_sha256": sha256(media_manifest) if media_manifest else None,
        "blind_judgments_immutable_in_later_phases": True,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }
    summary_path = out.with_suffix(out.suffix + ".freeze.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(
        "instructional_blind", "instructional_label", "witnessed_blind",
        "commentary_blind", "commentary_label", "commentary_model_reveal",
    ), required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--prior-freeze", type=Path)
    parser.add_argument("--media-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(
        args.phase, args.cohort, args.ledger, args.out, args.prior_freeze,
        args.media_manifest,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
