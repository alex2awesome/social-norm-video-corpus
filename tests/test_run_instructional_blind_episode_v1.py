import json

from scripts.run_instructional_blind_episode_v1 import (
    SYSTEM_BLIND_EPISODE, parse_blind_episode, quotes_by_item,
)


def result(**changes):
    value = {
        "episode_medium": "roleplay",
        "visual_context": "roleplay_episode",
        "visible_participants": "employee and customer",
        "literal_visible_action": "employee points at customer",
        "quote_speech_act": "employee insults customer",
        "quote_function": "situated_to_present_party",
        "participant_grounding": "affected_party_present",
        "physical_social_action": "yes",
        "episode_complete": "yes",
        "evidence": "The employee addresses the visible customer.",
    }
    value.update(changes); return json.dumps(value)


def test_roleplay_and_same_actor_formats_are_not_excluded() -> None:
    assert parse_blind_episode(result())["episode_pass"] is True
    assert "one actor playing multiple roles" in SYSTEM_BLIND_EPISODE


def test_presenter_and_report_fail_even_with_social_words() -> None:
    parsed = parse_blind_episode(result(
        visual_context="presenter_or_interview",
        participant_grounding="actor_only",
        quote_function="presenter_advice",
    ))
    assert parsed["episode_pass"] is False


def test_technical_demo_fails_without_affected_party() -> None:
    parsed = parse_blind_episode(result(
        episode_medium="technical_or_solo_demo",
        visual_context="technical_or_solo_demo",
        participant_grounding="none", physical_social_action="no",
        quote_function="narration_or_report",
    ))
    assert parsed["episode_pass"] is False


def test_quotes_context_cannot_leak_norm_or_polarity() -> None:
    context = quotes_by_item([{
        "item_id": "a", "start_quote": "hello", "end_quote": "goodbye",
        "norm": "respect", "polarity": "violation", "title": "leading",
    }])["a"]
    assert context == {"start_quote": "hello", "end_quote": "goodbye"}
