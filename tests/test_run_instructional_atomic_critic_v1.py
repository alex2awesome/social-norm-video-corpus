import pytest

from scripts.run_instructional_atomic_critic_v1 import (
    SYSTEM_ATOMIC,
    parse_atomic,
    relation_matches_polarity,
)


def valid_result(**changes: str) -> str:
    values = {
        "episode_medium": "roleplay",
        "actor_action": "employee insults a customer",
        "affected_party": "customer standing at the counter",
        "grounding": "situated_speech_to_present_party",
        "episode_continuity": "connected",
        "social_basis": "institutional_fairness",
        "relation": "violates",
        "completeness": "complete",
        "evidence": "The employee directs the insult at the visible customer.",
    }
    values.update(changes)
    import json
    return json.dumps(values)


def test_roleplay_situated_speech_can_pass() -> None:
    result = parse_atomic(valid_result(), "violation")
    assert result["atomic_pass"] is True
    assert result["relation_matches_proposed_polarity"] is True


def test_observed_relation_not_metadata_controls_polarity() -> None:
    result = parse_atomic(valid_result(relation="complies"), "violation")
    assert result["atomic_pass"] is False
    assert relation_matches_polarity("complies", "correct") is True


def test_technical_and_self_only_episodes_fail_social_scope() -> None:
    technical = parse_atomic(
        valid_result(social_basis="technical_procedure", relation="complies"),
        "correct",
    )
    self_only = parse_atomic(
        valid_result(social_basis="self_only", relation="complies"), "correct"
    )
    assert technical["atomic_pass"] is False
    assert self_only["atomic_pass"] is False


def test_presenter_or_metadata_only_fails_episode_grounding() -> None:
    presenter = parse_atomic(
        valid_result(
            grounding="metadata_or_narration_only",
            episode_continuity="presenter_or_interview",
        ),
        "violation",
    )
    assert presenter["atomic_pass"] is False


def test_explicit_no_episode_is_a_valid_fail_closed_output() -> None:
    result = parse_atomic(valid_result(episode_continuity="none"), "violation")
    assert result["atomic_pass"] is False


def test_explicit_no_grounding_is_a_valid_fail_closed_output() -> None:
    result = parse_atomic(valid_result(grounding="none"), "violation")
    assert result["atomic_pass"] is False


@pytest.mark.parametrize("relation", ["complies", "violates", "contrast", "illustrates"])
def test_explanation_accepts_any_concrete_related_direction(relation: str) -> None:
    assert parse_atomic(valid_result(relation=relation), "explanation")["atomic_pass"]


def test_contrast_requires_both_sides_or_explicit_correction() -> None:
    assert parse_atomic(valid_result(relation="contrast"), "contrast")["atomic_pass"]
    assert not parse_atomic(valid_result(relation="violates"), "contrast")["atomic_pass"]


def test_schema_is_strict() -> None:
    with pytest.raises(ValueError, match="schema mismatch"):
        parse_atomic(valid_result().replace("}", ', "is_social_norm": "yes"}'), "violation")


def test_prompt_operationalizes_social_scope_and_valid_media() -> None:
    assert "Do not make" in SYSTEM_ATOMIC
    assert "one actor playing multiple roles" in SYSTEM_ATOMIC
    assert "institutional_fairness" in SYSTEM_ATOMIC
    assert "self_only" in SYSTEM_ATOMIC
    assert "Do not call a correct act a violation" in SYSTEM_ATOMIC
