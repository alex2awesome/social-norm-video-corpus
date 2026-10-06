import json

import pytest

from scripts.assign_instructional_evidence_bands import assign_band
from scripts.score_blind_event_alignment import RUBRIC_VERSION_V2, parse_json


def observer(event: bool = True) -> dict:
    return {
        "result": {
            "visually_observable_event": "yes" if event else "no",
            "event_start_percent": 10 if event else -1,
            "event_end_percent": 50 if event else -1,
            "actor_visible_description": "a child" if event else "none",
            "action_or_situated_speech_description": (
                "hands over a ball" if event else "none"
            ),
            "affected_party_or_shared_setting_description": (
                "another child" if event else "none"
            ),
            "evidence_before": "first child holds ball" if event else "none",
            "evidence_during": "both touch ball" if event else "none",
            "evidence_after": "second child holds ball" if event else "none",
            "metadata_needed_to_identify_action": "no",
            "presentation_or_context_only": "no" if event else "yes",
            "depiction_type": "enacted_scene",
            "confidence": 0.9,
            "evidence": "The ball transfers." if event else "No event.",
        }
    }


def alignment(exact: str = "yes", usable: str = "yes") -> dict:
    return {
        "result": {
            "both_observers_found_event": "yes",
            "observers_describe_same_event": "yes",
            "intersecting_actor": "a child",
            "intersecting_action": "hands over a ball",
            "intersecting_target_or_setting": "another child",
            "proposed_norm_supported_by_intersection": exact,
            "exact_label_supported": exact,
            "usable_after_relabel": usable,
            "normalized_behavior": "sharing a ball",
            "visual_evidence_independent_of_metadata": "yes",
            "failure_reason": "none",
            "reason": "Both observers describe the same transfer.",
        }
    }


def test_alignment_schema_requires_literal_intersection():
    result = alignment()["result"]
    assert parse_json(json.dumps(result)) == result

    del result["intersecting_action"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_json(json.dumps(result))


def test_v2_alignment_schema_separates_scene_from_label_and_domain():
    result = {
        "both_observers_found_event": "yes",
        "observers_describe_same_event": "yes",
        "intersecting_actor": "people in a crowd",
        "intersecting_action": "hold flags at a rally",
        "intersecting_target_or_setting": "a public political rally",
        "literal_social_behavior_visible": "no",
        "social_norm_domain_of_intersection": "political_or_formal",
        "proposed_label_kind": "political_or_formal",
        "proposed_label_action_visible": "yes",
        "response_or_aftermath_only_for_proposed_label": "no",
        "exact_label_supported": "no",
        "usable_after_relabel": "no",
        "normalized_behavior": "none",
        "visual_evidence_independent_of_metadata": "yes",
        "failure_reason": "political_or_formal",
        "reason": "The window shows political participation, not a tacit norm event.",
    }
    assert (
        parse_json(json.dumps(result), RUBRIC_VERSION_V2)
        == result
    )


def test_v2_alignment_schema_rejects_unrecognized_domains():
    result = {
        "both_observers_found_event": "yes",
        "observers_describe_same_event": "yes",
        "intersecting_actor": "an employee",
        "intersecting_action": "runs through a kitchen",
        "intersecting_target_or_setting": "a restaurant kitchen",
        "literal_social_behavior_visible": "no",
        "social_norm_domain_of_intersection": "generic_motion",
        "proposed_label_kind": "concrete_conduct",
        "proposed_label_action_visible": "no",
        "response_or_aftermath_only_for_proposed_label": "yes",
        "exact_label_supported": "no",
        "usable_after_relabel": "no",
        "normalized_behavior": "none",
        "visual_evidence_independent_of_metadata": "yes",
        "failure_reason": "response_or_aftermath_only",
        "reason": "The alleged attack is absent.",
    }
    with pytest.raises(ValueError, match="invalid v2"):
        parse_json(json.dumps(result), RUBRIC_VERSION_V2)


def test_v2_alignment_schema_rejects_contradictory_exact_label():
    result = {
        "both_observers_found_event": "yes",
        "observers_describe_same_event": "yes",
        "intersecting_actor": "a woman",
        "intersecting_action": "grabs another woman",
        "intersecting_target_or_setting": "another woman on a beach",
        "literal_social_behavior_visible": "yes",
        "social_norm_domain_of_intersection": "tacit_interpersonal",
        "proposed_label_kind": "concrete_conduct",
        "proposed_label_action_visible": "yes",
        "response_or_aftermath_only_for_proposed_label": "no",
        "exact_label_supported": "yes",
        "usable_after_relabel": "no",
        "normalized_behavior": "grabbing another person",
        "visual_evidence_independent_of_metadata": "yes",
        "failure_reason": "no_event",
        "reason": "The reason contradicts the exact-label field.",
    }
    with pytest.raises(ValueError, match="inconsistent v2 exact label"):
        parse_json(json.dumps(result), RUBRIC_VERSION_V2)


def test_v2_alignment_schema_rejects_recovery_without_social_event():
    result = {
        "both_observers_found_event": "yes",
        "observers_describe_same_event": "yes",
        "intersecting_actor": "people",
        "intersecting_action": "hold flags",
        "intersecting_target_or_setting": "a political rally",
        "literal_social_behavior_visible": "no",
        "social_norm_domain_of_intersection": "political_or_formal",
        "proposed_label_kind": "political_or_formal",
        "proposed_label_action_visible": "yes",
        "response_or_aftermath_only_for_proposed_label": "no",
        "exact_label_supported": "no",
        "usable_after_relabel": "yes",
        "normalized_behavior": "holding flags",
        "visual_evidence_independent_of_metadata": "yes",
        "failure_reason": "political_or_formal",
        "reason": "Political participation is not a tacit social event.",
    }
    with pytest.raises(ValueError, match="inconsistent v2 recovery"):
        parse_json(json.dumps(result), RUBRIC_VERSION_V2)


def test_exact_and_relabel_candidates_are_separate():
    assert (
        assign_band(observer(), observer(), alignment())
        == "blind_dual_exact_candidate"
    )
    assert (
        assign_band(observer(), observer(), alignment(exact="no"))
        == "blind_dual_relabel_candidate"
    )


def test_metadata_dependent_observation_cannot_enter_candidate_band():
    first = observer()
    first["result"]["metadata_needed_to_identify_action"] = "yes"

    assert (
        assign_band(first, observer(), alignment())
        == "single_blind_event_review"
    )


def test_observer_disagreement_is_review_only():
    result = alignment()
    result["result"]["observers_describe_same_event"] = "no"

    assert (
        assign_band(observer(), observer(), result)
        == "blind_event_semantic_conflict_review"
    )
