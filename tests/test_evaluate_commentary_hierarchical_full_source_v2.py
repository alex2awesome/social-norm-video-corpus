import copy

import pytest

from scripts.evaluate_commentary_hierarchical_full_source_v2 import evaluate


def fixture():
    selected = [{
        "window_id": "w", "uid": "u", "selection_reasons": ["feature_motion"]
    }]
    blind = [{
        "window_id": "w", "uid": "u", "performed_event_visible": "yes",
        "actor_target_grounded": "yes", "before_action_after_complete": "yes",
        "crucial_action_occluded_or_offframe": "no", "label_bearing_text_absent": "yes",
        "literal_action_description": "One person takes another's bag.",
        "manual_rationale": "The transfer is visible.",
        "source_audio_review_status": "reviewed",
        "event_evidence_modalities": "audiovisual_other",
    }]
    post = [{
        "window_id": "w", "uid": "u", "exact_named_action_visible": "yes",
        "label_alignment": "exact", "start_boundary_clean": "yes",
        "end_boundary_clean": "yes", "commentary_visual_route": "commentary_visual",
        "vlm_output_visually_supported": "yes", "corrected_behavior_label": "",
        "manual_rationale": "Exact and clean.",
    }]
    model = [{"candidate_id": "w", "model": "m", "strict_pass": True, "error": None}]
    review = [{
        "audit_index": "0", "item_id": "w", "uid": "u", "model": "m",
        "model_strict_pass": "yes", "structured_prediction_supported": "yes",
        "rationale_supported": "yes", "input_modality_claim_supported": "yes",
        "stage_a_literal_analysis_supported": "yes",
        "stage_b_alignment_supported_by_stage_a": "yes", "manual_rationale": "Supported.",
    }]
    media = [{"window_id": "w", "audio_present": True}]
    return selected, blind, post, model, review, media


def test_reports_window_and_source_metrics_without_claiming_source_recall():
    report = evaluate(*fixture())
    assert report["strict_exact_window_metrics"]["fail_closed"]["tp"] == 1
    assert report["source_has_strict_exact_selected_window_metrics"]["fail_closed"]["tp"] == 1
    assert report["source_localization_recall_estimated"] is False
    assert report["automatic_acceptance"] is False
    assert report["requires_muted_exact_artifact_materialization_and_reaudit"] is True
    assert report["individual_exact_clips_accepted"] == 0


def test_label_text_leak_fails_strict_exact_contract():
    selected, blind, post, model, review, media = fixture()
    blind[0]["label_bearing_text_absent"] = "no"
    report = evaluate(selected, blind, post, model, review, media)
    assert report["strict_exact_window_metrics"]["fail_closed"]["fp"] == 1


def test_requires_every_model_rationale_reviewed():
    selected, blind, post, model, review, media = fixture()
    bad = copy.deepcopy(review)
    bad[0]["stage_b_alignment_supported_by_stage_a"] = ""
    with pytest.raises(ValueError, match="stage_b"):
        evaluate(selected, blind, post, model, bad, media)


def test_rejects_changed_copied_model_prediction():
    selected, blind, post, model, review, media = fixture()
    review[0]["model_strict_pass"] = "no"
    with pytest.raises(ValueError, match="prediction changed"):
        evaluate(selected, blind, post, model, review, media)


def test_relabel_requires_concrete_corrected_behavior():
    selected, blind, post, model, review, media = fixture()
    post[0].update({
        "exact_named_action_visible": "no", "label_alignment": "partial",
        "commentary_visual_route": "relabel_required",
    })
    with pytest.raises(ValueError, match="lacks corrected"):
        evaluate(selected, blind, post, model, review, media)
