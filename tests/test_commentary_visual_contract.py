from __future__ import annotations

from scripts.commentary_visual_contract import derive_commentary_visual_disposition


def candidate(**overrides):
    row = {
        "text_label_decision": "accept",
        "behavior_occurrence": "on_camera_visual_action",
        "event_identity_grounded": "yes",
        "actor_target_or_context_grounded_in_clip": "yes",
        "label_span_relation": "separate_after_behavior",
        "action_clip_contains_normative_label_signal": "no",
        "action_artifact_transform": "temporal_cut",
        "sanitization_verified": "yes",
        "demo_quality": "clear_visual",
        "boundary_basis": "exact_video",
        "authenticity": "organic",
        "action_start_sec": 1.0,
        "action_end_sec": 4.0,
    }
    row.update(overrides)
    return derive_commentary_visual_disposition(row)


def test_separate_text_label_can_supervise_exact_visual_occurrence():
    assert candidate()["disposition"] == "recover_commentary_visual"


def test_on_camera_speech_act_is_audiovisual_demo():
    result = candidate(
        behavior_occurrence="on_camera_speech_act",
        demo_quality="clear_audiovisual",
        label_span_relation="separate_before_behavior",
    )
    assert result["disposition"] == "recover_commentary_visual"


def test_actual_event_inside_news_wrapper_can_recover():
    result = candidate(
        behavior_occurrence="on_camera_speech_act",
        demo_quality="clear_audiovisual",
        label_span_relation="separate_before_behavior",
        authenticity="organic_news_footage",
    )
    assert result["disposition"] == "recover_commentary_visual"


def test_broll_description_remains_text_only():
    assert candidate(behavior_occurrence="described_over_broll")["disposition"] == "text_only"


def test_same_utterance_label_leak_remains_text_only():
    assert candidate(label_span_relation="same_utterance")["disposition"] == "text_only"


def test_overlapping_label_can_only_pass_after_verified_mute_and_crop():
    result = candidate(
        label_span_relation="overlaps_behavior",
        action_artifact_transform="mute_and_crop",
        sanitization_verified="yes",
    )
    assert result["disposition"] == "recover_commentary_visual"


def test_overlap_with_only_crop_remains_text_only():
    result = candidate(
        label_span_relation="overlaps_behavior",
        action_artifact_transform="crop",
        sanitization_verified="yes",
    )
    assert result["disposition"] == "text_only"


def test_unverified_transformation_is_uncertain():
    result = candidate(sanitization_verified="uncertain")
    assert result["disposition"] == "uncertain"


def test_action_clip_with_stance_leak_remains_text_only():
    assert candidate(action_clip_contains_normative_label_signal="yes")["disposition"] == "text_only"


def test_sparse_frames_cannot_certify_visual_recovery():
    assert candidate(boundary_basis="frame_grid_only")["disposition"] == "text_only"


def test_scripted_occurrence_goes_through_instructional_gate():
    assert candidate(authenticity="scripted")["disposition"] == "reroute_instructional_review"
