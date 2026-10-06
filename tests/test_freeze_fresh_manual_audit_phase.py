import csv
import json
from pathlib import Path

import pytest

from scripts.freeze_fresh_manual_audit_phase import freeze


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def instruction_row() -> dict[str, str]:
    return {
        "item_id": "i", "uid": "u", "visual_demo": "yes",
        "temporal_state_change": "yes",
        "recipient_response_or_coordinated_trajectory": "yes",
        "segment_start_frame": "2", "segment_end_frame": "8",
        "label_alignment": "", "manual_description": "One person hands an item.",
        "manual_note": "", "source_audio_review_status": "reviewed",
        "demo_evidence_modalities": "audiovisual_other",
    }


def instruction_media(tmp_path: Path) -> Path:
    path = tmp_path / "instruction_media.jsonl"
    write_jsonl(path, [{"item_id": "i", "uid": "u", "audio_present": True}])
    return path


def test_instruction_blind_then_label_freeze_locks_visual_gold(tmp_path: Path):
    cohort = tmp_path / "cohort.jsonl"
    ledger = tmp_path / "ledger.tsv"
    media = instruction_media(tmp_path)
    write_jsonl(cohort, [{"item_id": "i", "uid": "u"}])
    row = instruction_row()
    write_tsv(ledger, [row])
    blind = tmp_path / "blind.tsv"
    assert freeze(
        "instructional_blind", cohort, ledger, blind, media_manifest=media
    )["rows"] == 1

    row["label_alignment"] = "exact"
    write_tsv(ledger, [row])
    assert freeze(
        "instructional_label", cohort, ledger, tmp_path / "label.tsv", blind,
        media,
    )["prior_freeze_sha256"]


def test_instruction_label_rejects_changed_blind_judgment(tmp_path: Path):
    cohort = tmp_path / "cohort.jsonl"
    ledger = tmp_path / "ledger.tsv"
    media = instruction_media(tmp_path)
    write_jsonl(cohort, [{"item_id": "i", "uid": "u"}])
    row = instruction_row()
    write_tsv(ledger, [row])
    blind = tmp_path / "blind.tsv"
    freeze("instructional_blind", cohort, ledger, blind, media_manifest=media)
    row.update({"visual_demo": "no", "segment_start_frame": "-1", "segment_end_frame": "-1", "label_alignment": "no_visual", "demo_evidence_modalities": "no_demo"})
    write_tsv(ledger, [row])
    with pytest.raises(ValueError, match="frozen field changed"):
        freeze("instructional_label", cohort, ledger, tmp_path / "label.tsv", blind, media)


def test_instruction_partial_alignment_requires_explicit_relabel(tmp_path: Path):
    cohort = tmp_path / "cohort.jsonl"
    ledger = tmp_path / "ledger.tsv"
    media = instruction_media(tmp_path)
    write_jsonl(cohort, [{"item_id": "i", "uid": "u"}])
    row = {**instruction_row(), "corrected_behavior_label": ""}
    write_tsv(ledger, [row])
    blind = tmp_path / "blind.tsv"
    freeze("instructional_blind", cohort, ledger, blind, media_manifest=media)
    row["label_alignment"] = "partial"
    write_tsv(ledger, [row])
    with pytest.raises(ValueError, match="requires corrected_behavior_label"):
        freeze("instructional_label", cohort, ledger, tmp_path / "label.tsv", blind, media)


def test_instruction_partial_alignment_can_be_frozen_after_blind_pass(tmp_path: Path):
    cohort = tmp_path / "cohort.jsonl"
    ledger = tmp_path / "ledger.tsv"
    media = instruction_media(tmp_path)
    write_jsonl(cohort, [{"item_id": "i", "uid": "u"}])
    row = {**instruction_row(), "corrected_behavior_label": "person returns item"}
    row["corrected_behavior_label"] = ""
    write_tsv(ledger, [row])
    blind = tmp_path / "blind.tsv"
    freeze("instructional_blind", cohort, ledger, blind, media_manifest=media)
    row.update({
        "label_alignment": "partial",
        "corrected_behavior_label": "person returns another person's item",
    })
    write_tsv(ledger, [row])
    assert freeze(
        "instructional_label", cohort, ledger, tmp_path / "label.tsv", blind,
        media,
    )["rows"] == 1


def test_instruction_no_visual_requires_no_visual_alignment(tmp_path: Path):
    cohort = tmp_path / "cohort.jsonl"
    ledger = tmp_path / "ledger.tsv"
    media = instruction_media(tmp_path)
    write_jsonl(cohort, [{"item_id": "i", "uid": "u"}])
    row = {**instruction_row(), "corrected_behavior_label": ""}
    write_tsv(ledger, [row])
    blind = tmp_path / "blind.tsv"
    freeze("instructional_blind", cohort, ledger, blind, media_manifest=media)
    row["label_alignment"] = "no_visual"
    write_tsv(ledger, [row])
    with pytest.raises(ValueError, match="contradict"):
        freeze("instructional_label", cohort, ledger, tmp_path / "label.tsv", blind, media)


def commentary_post_row() -> dict[str, str]:
    return {
        "window_id": "c", "uid": "u", "exact_named_action_visible": "yes",
        "label_alignment": "exact", "start_boundary_clean": "yes",
        "end_boundary_clean": "yes", "commentary_visual_route": "commentary_visual",
        "vlm_output_visually_supported": "", "manual_rationale": "Exact visible act.",
    }


def test_commentary_blind_freeze_requires_available_audio_review(tmp_path: Path):
    cohort = tmp_path / "cohort.jsonl"
    media = tmp_path / "media.jsonl"
    ledger = tmp_path / "blind.tsv"
    write_jsonl(cohort, [{"window_id": "c", "uid": "u"}])
    write_jsonl(media, [{"window_id": "c", "audio_present": True}])
    row = {
        "window_id": "c", "uid": "u", "performed_event_visible": "yes",
        "actor_target_grounded": "yes", "before_action_after_complete": "yes",
        "crucial_action_occluded_or_offframe": "no",
        "label_bearing_text_absent": "yes",
        "literal_action_description": "One person apologizes to another.",
        "manual_rationale": "The depicted speaker addresses the recipient.",
        "source_audio_review_status": "reviewed",
        "event_evidence_modalities": "audiovisual_speech_act",
    }
    write_tsv(ledger, [row])
    assert freeze(
        "commentary_blind", cohort, ledger, tmp_path / "frozen.tsv",
        media_manifest=media,
    )["rows"] == 1
    row["source_audio_review_status"] = "source_has_no_audio"
    write_tsv(ledger, [row])
    with pytest.raises(ValueError, match="not completely reviewed"):
        freeze(
            "commentary_blind", cohort, ledger, tmp_path / "bad.tsv",
            media_manifest=media,
        )


def test_commentary_label_must_precede_model_support_reveal(tmp_path: Path):
    cohort = tmp_path / "cohort.jsonl"
    ledger = tmp_path / "post.tsv"
    write_jsonl(cohort, [{"window_id": "c", "uid": "u"}])
    row = commentary_post_row()
    write_tsv(ledger, [row])
    label = tmp_path / "label.tsv"
    freeze("commentary_label", cohort, ledger, label)
    row["vlm_output_visually_supported"] = "yes"
    write_tsv(ledger, [row])
    assert freeze(
        "commentary_model_reveal", cohort, ledger, tmp_path / "model.tsv", label
    )["rows"] == 1


def test_commentary_label_rejects_early_model_reveal(tmp_path: Path):
    cohort = tmp_path / "cohort.jsonl"
    ledger = tmp_path / "post.tsv"
    write_jsonl(cohort, [{"window_id": "c", "uid": "u"}])
    row = commentary_post_row()
    row["vlm_output_visually_supported"] = "yes"
    write_tsv(ledger, [row])
    with pytest.raises(ValueError, match="revealed during label phase"):
        freeze("commentary_label", cohort, ledger, tmp_path / "label.tsv")


def witnessed_row() -> dict[str, str]:
    return {
        "candidate_id": "w", "item_id": "clip", "uid": "u",
        "reaction_grounded": "yes",
        "action_before_or_overlaps_response": "yes",
        "response_targets_action": "yes",
        "responder_role": "separate_bystander",
        "response_content": "targeted_objection",
        "trigger_kind": "interpersonal_treatment",
        "staging": "no_clear_staging_evidence",
        "manual_evidence": "A visible third person objects after the act.",
        "manual_note": "", "source_audio_review_status": "reviewed",
        "speaker_identity_basis": "voice_and_visible_turn_binding",
    }


def test_witnessed_freeze_requires_audio_review_and_positive_speaker_binding(tmp_path: Path):
    cohort = tmp_path / "cohort.jsonl"
    media = tmp_path / "media.jsonl"
    ledger = tmp_path / "ledger.tsv"
    write_jsonl(cohort, [{
        "item_id": "clip", "uid": "u",
        "candidates": [{"candidate_id": "w"}],
    }])
    write_jsonl(media, [{"candidate_id": "w", "audio_present": True}])
    row = witnessed_row()
    write_tsv(ledger, [row])
    assert freeze(
        "witnessed_blind", cohort, ledger, tmp_path / "frozen.tsv",
        media_manifest=media,
    )["media_manifest_sha256"]
    row["speaker_identity_basis"] = "offscreen_or_unresolved"
    write_tsv(ledger, [row])
    with pytest.raises(ValueError, match="lacks positive speaker binding"):
        freeze(
            "witnessed_blind", cohort, ledger, tmp_path / "bad.tsv",
            media_manifest=media,
        )


def test_witnessed_freeze_rejects_skipped_available_audio(tmp_path: Path):
    cohort = tmp_path / "cohort.jsonl"
    media = tmp_path / "media.jsonl"
    ledger = tmp_path / "ledger.tsv"
    write_jsonl(cohort, [{
        "item_id": "clip", "uid": "u",
        "candidates": [{"candidate_id": "w"}],
    }])
    write_jsonl(media, [{"candidate_id": "w", "audio_present": True}])
    row = witnessed_row(); row["source_audio_review_status"] = "source_has_no_audio"
    write_tsv(ledger, [row])
    with pytest.raises(ValueError, match="not completely reviewed"):
        freeze(
            "witnessed_blind", cohort, ledger, tmp_path / "bad.tsv",
            media_manifest=media,
        )
