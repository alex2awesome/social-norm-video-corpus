import json

import pytest

from scripts.run_instructional_temporal_critic_v4 import (
    derive_demo_candidate,
    parse_temporal_critic,
)


def valid_result() -> dict:
    return {
        "segment_kind": "animated_episode",
        "segment_start_frame": 4,
        "segment_end_frame": 12,
        "recipient_configuration": "visible_recipient_or_affected_party",
        "temporal_act_evidence": "visible_before_act_after",
        "specific_social_act": "yes",
        "generic_conversation_or_activity": "no",
        "static_meaning_only": "no",
        "presenter_or_program": "no",
        "disconnected_montage": "no",
        "technical_or_nonsocial": "no",
        "performed_act": "child returns an object",
        "before_evidence": "child holds the object",
        "during_evidence": "child hands it to another character",
        "after_evidence": "the other character holds it",
        "evidence": "The object visibly changes hands in one episode.",
    }


def test_temporal_critic_accepts_visible_state_change() -> None:
    parsed = parse_temporal_critic(json.dumps(valid_result()))
    assert parsed["demo_candidate"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temporal_act_evidence", "clear_static_tableau_no_motion"),
        ("generic_conversation_or_activity", "yes"),
        ("static_meaning_only", "yes"),
        ("presenter_or_program", "yes"),
        ("disconnected_montage", "yes"),
        ("technical_or_nonsocial", "yes"),
    ],
)
def test_temporal_critic_vetoes_known_false_positive_mechanisms(field, value) -> None:
    result = valid_result()
    result[field] = value
    assert derive_demo_candidate(result) is False


def test_temporal_critic_accepts_screen_action_and_public_coordination() -> None:
    for kind, recipient in [
        ("screen_action_episode", "visible_recipient_or_affected_party"),
        ("public_coordination_episode", "coordinating_agents_or_vehicles"),
    ]:
        result = valid_result()
        result["segment_kind"] = kind
        result["recipient_configuration"] = recipient
        assert derive_demo_candidate(result) is True


def test_temporal_critic_rejects_invalid_or_partial_frame_bounds() -> None:
    result = valid_result()
    result["segment_start_frame"] = -1
    with pytest.raises(ValueError, match="both be -1"):
        parse_temporal_critic(json.dumps(result))
    result = valid_result()
    result["segment_start_frame"] = 20
    result["segment_end_frame"] = 10
    with pytest.raises(ValueError, match="start exceeds end"):
        parse_temporal_critic(json.dumps(result))
