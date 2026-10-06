import pytest

from scripts.run_instructional_demo_adjudicator_v3 import (
    derive_demo_candidate,
    parse_demo_adjudicator,
)


def result(**overrides: str) -> dict[str, str]:
    row = {
        "visual_sequence": "continuous_episode",
        "participant_configuration": "visible_interacting_participants",
        "behavior_evidence": "visibly_performed",
        "specific_social_act": "yes",
        "presenter_or_program": "no",
        "disconnected_broll": "no",
        "technical_or_generic_activity": "no",
        "literal_behavior": "One person apologizes to another.",
        "evidence": "Both parties and the apology are visible in one episode.",
    }
    row.update(overrides)
    return row


def test_accepts_connected_social_episode() -> None:
    assert derive_demo_candidate(result()) is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("visual_sequence", "presenter_or_interview"),
        ("participant_configuration", "depicted_people_without_interaction"),
        ("behavior_evidence", "only_spoken_or_inferred"),
        ("specific_social_act", "no"),
        ("presenter_or_program", "yes"),
        ("disconnected_broll", "yes"),
        ("technical_or_generic_activity", "yes"),
    ],
)
def test_each_failure_atom_vetoes(field: str, value: str) -> None:
    assert derive_demo_candidate(result(**{field: value})) is False


def test_accepts_caregiving_and_solo_convention_when_visually_grounded() -> None:
    caregiving = result(literal_behavior="Doctor comforts and examines patient.")
    solo = result(
        visual_sequence="solo_convention",
        participant_configuration="recognizable_solo_convention",
        literal_behavior="Child descends a playground slide feet first.",
    )
    assert derive_demo_candidate(caregiving) is True
    assert derive_demo_candidate(solo) is True


def test_parser_is_exact_and_materializes_decision() -> None:
    import json

    parsed = parse_demo_adjudicator(json.dumps(result()))
    assert parsed["demo_candidate"] is True
    with pytest.raises(ValueError, match="schema mismatch"):
        parse_demo_adjudicator(json.dumps({**result(), "extra": "bad"}))
