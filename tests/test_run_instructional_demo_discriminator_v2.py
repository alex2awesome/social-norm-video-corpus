import json

import pytest

from scripts.run_instructional_demo_discriminator_v2 import (
    derive_demo_candidate,
    parse_demo_discriminator,
)


def result(**changes: str) -> dict[str, str]:
    row = {
        "depiction_medium": "live_action",
        "depiction_mode": "captured_or_fictional_episode",
        "behavior_source": "visibly_performed",
        "social_act_kind": "situated_social_speech",
        "recipient_grounding": "visible_recipient",
        "episode_continuity": "same_episode",
        "interview_or_presentation": "no",
        "technical_skill": "no",
        "concrete_visible_behavior": "one person refuses another person's request",
        "evidence": "Both people share the episode and respond to each other.",
    }
    row.update(changes)
    return row


def test_episode_phone_solo_etiquette_and_static_tableau_can_pass() -> None:
    assert derive_demo_candidate(result()) is True
    assert derive_demo_candidate(result(
        recipient_grounding="offscreen_recipient_same_episode",
    )) is True
    assert derive_demo_candidate(result(
        depiction_mode="solo_etiquette_demonstration",
        social_act_kind="solo_social_convention",
        recipient_grounding="solo_convention",
    )) is True
    assert derive_demo_candidate(result(
        depiction_medium="synthetic_or_static",
        depiction_mode="static_behavior_tableau",
        behavior_source="visibly_static_depiction",
        episode_continuity="clear_single_tableau",
    )) is True


@pytest.mark.parametrize(
    "changes",
    [
        {"depiction_mode": "interview_or_conversation_program",
         "interview_or_presentation": "yes"},
        {"depiction_mode": "presenter_with_disconnected_broll",
         "episode_continuity": "disconnected_or_presenter_broll"},
        {"depiction_mode": "technical_skill_demonstration",
         "social_act_kind": "technical_non_social_action", "technical_skill": "yes"},
        {"behavior_source": "only_spoken_or_described"},
        {"social_act_kind": "generic_conversation"},
        {"recipient_grounding": "none"},
    ],
)
def test_known_transfer_false_positive_modes_fail(changes: dict[str, str]) -> None:
    assert derive_demo_candidate(result(**changes)) is False


def test_parser_is_exact_and_derives_decision() -> None:
    parsed = parse_demo_discriminator(json.dumps(result()))
    assert parsed["demo_candidate"] is True
    malformed = result()
    malformed["unexpected"] = "field"
    with pytest.raises(ValueError, match="schema mismatch"):
        parse_demo_discriminator(json.dumps(malformed))


def test_uncertain_values_abstain() -> None:
    assert derive_demo_candidate(result(technical_skill="unclear")) is False
