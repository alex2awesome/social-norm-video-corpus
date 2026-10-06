from __future__ import annotations

import pytest

from scripts.instructional_demo_contract import (
    derive_instructional_demo_disposition,
)


def candidate(**overrides):
    row = {
        "actor_kind": "person",
        "behavior_kind": "physical_action",
        "affected_context_kind": "person",
        "expectation_kind": "interpersonal_treatment",
        "performed_behavior": "yes",
        "actor_grounded_in_demo": "yes",
        "target_or_social_context_grounded_in_demo": "yes",
        "connected_episode": "yes",
        "label_alignment": "exact",
        "demo_medium": "live_action",
        "demo_quality": "clear_visual",
        "boundary_basis": "exact_video",
        "polarity": "violation",
        "demo_start_sec": 1.0,
        "demo_end_sec": 5.0,
    }
    row.update(overrides)
    return derive_instructional_demo_disposition(row)


@pytest.mark.parametrize(
    "medium",
    [
        "live_action",
        "animation",
        "puppet_or_toy",
        "game_or_simulation",
        "hidden_camera_or_social_experiment",
    ],
)
def test_any_performed_medium_can_be_a_demo(medium):
    assert candidate(demo_medium=medium)["disposition"] == (
        "instructional_demo_candidate"
    )


@pytest.mark.parametrize(
    "polarity", ["violation", "correct", "contrast", "explanation"]
)
def test_polarity_does_not_replace_observed_demo_atoms(polarity):
    assert candidate(polarity=polarity)["disposition"] == (
        "instructional_demo_candidate"
    )


@pytest.mark.parametrize(
    "medium",
    [
        "talking_head_only",
        "text_or_graphic_only",
        "generic_broll",
        "screen_tutorial",
        "audio_only",
    ],
)
def test_nonperformed_formats_are_text_only(medium):
    assert candidate(demo_medium=medium)["disposition"] == (
        "text_only_instructional"
    )


def test_offscreen_description_is_not_a_visual_demo():
    assert candidate(performed_behavior="no")["disposition"] == (
        "text_only_instructional"
    )


def test_clear_scene_with_wrong_label_is_preserved_for_relabel():
    assert candidate(label_alignment="mismatch")["disposition"] == (
        "instructional_demo_candidate_after_relabel"
    )


def test_frame_grid_can_propose_recut_but_not_certify_clip():
    assert candidate(
        boundary_basis="frame_grid_only",
        demo_start_sec=None,
        demo_end_sec=None,
    )["disposition"] == "instructional_source_needs_recut"


def test_out_of_scope_technical_demo_does_not_enter_social_demo_route():
    assert candidate(
        behavior_kind="technical_procedure",
        demo_medium="screen_tutorial",
    )["disposition"] == "out_of_scope_review"


def test_uncertain_atomic_evidence_fails_closed():
    assert candidate(connected_episode="uncertain")["disposition"] == (
        "uncertain"
    )


def test_invalid_bounds_cannot_certify_exact_demo():
    assert candidate(demo_start_sec=5.0, demo_end_sec=4.0)["disposition"] == (
        "instructional_source_needs_recut"
    )


def test_invalid_enum_is_rejected():
    with pytest.raises(ValueError, match="invalid demo_medium"):
        candidate(demo_medium="documentary-ish")

