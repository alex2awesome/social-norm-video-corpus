from __future__ import annotations

from scripts.witnessed_signal_contract import derive_witnessed_disposition


def candidate(**overrides):
    row = {
        "actor_kind": "person",
        "behavior_kind": "speech_act",
        "affected_context_kind": "person",
        "expectation_kind": "interpersonal_treatment",
        "action_observed": "yes",
        "reaction_source_role": "bystander",
        "reaction_grounding": "audible_on_scene",
        "reaction_content": "targeted_objection",
        "temporal_relation": "action_established_before_reaction",
        "reaction_candidate_origin": "detector_selected",
        "current_label_relation": "exact",
        "pre_reaction_demo_quality": "clear_audiovisual",
        "boundary_basis": "exact_video",
        "authenticity": "organic",
        "source_staging_title_cue": False,
        "authority_reaction_text_cue": False,
        "action_end_sec": 4.0,
        "reaction_start_sec": 4.1,
    }
    row.update(overrides)
    return derive_witnessed_disposition(row)


def test_current_grounded_signal_can_strictly_accept():
    assert candidate()["disposition"] == "strict_accept"


def test_nearby_reaction_is_recovery_not_existing_item_acceptance():
    assert candidate(reaction_candidate_origin="nearby_transcript_recovery")["disposition"] == "recover_strict"


def test_repairable_label_requires_recovery():
    assert candidate(current_label_relation="repairable")["disposition"] == "recover_strict"


def test_direct_target_objection_does_not_expand_witnessed_signal():
    assert candidate(reaction_source_role="affected_target")["disposition"] == "reject"


def test_social_bystander_interposition_is_supported():
    assert candidate(
        reaction_content="interposition_or_separation"
    )["disposition"] == "strict_accept"


def test_safety_only_interposition_still_fails_social_scope():
    assert candidate(
        reaction_content="interposition_or_separation",
        expectation_kind="health_or_physical_safety_only",
    )["disposition"] == "reject"


def test_authority_command_is_preserved_as_scene_candidate_not_witnessed():
    result = candidate(reaction_source_role="authority_or_host")
    assert result["disposition"] == "scene_candidate"
    assert (
        result["strict_witnessed_exclusion_reason"]
        == "authority_or_host_reaction"
    )


def test_authority_command_without_clear_demo_is_rejected():
    assert (
        candidate(
            reaction_source_role="authority_or_host",
            pre_reaction_demo_quality="incomplete_or_ambiguous",
        )["disposition"]
        == "reject"
    )


def test_confirmed_authority_text_cue_preserves_scene_but_excludes_witnessed():
    result = candidate(authority_reaction_text_cue=True)
    assert result["disposition"] == "scene_candidate"
    assert (
        result["strict_witnessed_exclusion_reason"]
        == "authority_reaction_text_cue_v3"
    )


def test_confirmed_authority_text_cue_without_scene_is_rejected():
    assert (
        candidate(
            authority_reaction_text_cue=True,
            pre_reaction_demo_quality="none",
        )["disposition"]
        == "reject"
    )


def test_authority_text_cue_is_type_checked():
    try:
        candidate(authority_reaction_text_cue="yes")
    except ValueError as exc:
        assert "authority_reaction_text_cue" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_voiceover_does_not_count_as_on_scene_reaction():
    assert candidate(reaction_grounding="voiceover_or_remote")["disposition"] == "reject"


def test_frame_grid_cannot_certify_splice():
    assert candidate(boundary_basis="frame_grid_only")["disposition"] == "reject"


def test_safety_only_event_is_out_of_scope_even_with_reaction():
    assert candidate(expectation_kind="health_or_physical_safety_only")["disposition"] == "reject"


def test_scripted_demo_is_only_rerouted_for_independent_instructional_review():
    result = candidate(
        authenticity="scripted",
        boundary_basis="frame_grid_only",
        action_end_sec=None,
        reaction_start_sec=None,
    )
    assert result["disposition"] == "reroute_instructional_review"


def test_ambiguous_scripted_action_is_rejected():
    result = candidate(
        authenticity="scripted",
        pre_reaction_demo_quality="incomplete_or_ambiguous",
    )
    assert result["disposition"] == "reject"


def test_audited_prank_title_cannot_strictly_accept_as_organic_witnessed():
    result = candidate(source_staging_title_cue=True)
    assert result["disposition"] == "reroute_instructional_review"
    assert result["production_style"] == "staged_prank_candidate"
    assert (
        result["staging_rule_id"]
        == "title_prank_social_experiment_hidden_camera_v1"
    )


def test_staging_title_cue_is_type_checked():
    try:
        candidate(source_staging_title_cue="yes")
    except ValueError as exc:
        assert "source_staging_title_cue" in str(exc)
    else:
        raise AssertionError("expected ValueError")
