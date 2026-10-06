import copy

import pytest

from scripts.commentary_occurred_event_contract_v3 import validate_result


def clean():
    return {
        "is_social_norm": "yes",
        "occurred_event_grounded": "yes",
        "social_actor_grounded": "yes",
        "behavior_semantically_specific": "yes",
        "target_or_shared_context_grounded": "yes",
        "normative_stance_grounded": "yes",
        "event_scope": "bounded_occurrence",
        "stance_quality": "unambiguous_external",
        "behavior_evidence_quote": "the driver entered the bike lane",
        "stance_evidence_quote": "that is illegal and unsafe",
        "normalized_behavior": "a driver enters a bicycle lane",
        "normalized_norm": "keep vehicles out of bicycle lanes",
        "route": "strict_visual_search",
        "description": "One occurred maneuver and criticism are grounded.",
    }


def test_clean_occurred_event_enters_only_manual_visual_search():
    result = validate_result(clean())
    assert result["strict_event_candidate"] is True
    assert result["automatic_acceptance"] is False


def test_generic_definition_is_instructional_retrieval_not_event_label():
    row = clean()
    row.update(
        occurred_event_grounded="no",
        event_scope="generic_or_hypothetical",
        route="instructional_definition_retrieval",
    )
    result = validate_result(row)
    assert result["strict_event_candidate"] is False


def test_contested_reaction_cannot_be_unambiguous_positive():
    row = clean()
    row.update(stance_quality="contested", route="exploratory_contested")
    assert validate_result(row)["strict_event_candidate"] is False
    bad = copy.deepcopy(row)
    bad["route"] = "strict_visual_search"
    with pytest.raises(ValueError, match="strict route"):
        validate_result(bad)


def test_bare_evaluative_insult_is_not_specific_behavior():
    row = clean()
    row.update(
        behavior_semantically_specific="no",
        event_scope="ambiguous_or_unresolved",
        route="exploratory_ambiguous",
        normalized_behavior="",
        normalized_norm="",
    )
    assert validate_result(row)["strict_event_candidate"] is False


def test_aggregate_examples_are_exploratory_not_strict():
    row = clean()
    row.update(event_scope="aggregate_examples", route="exploratory_ambiguous")
    assert validate_result(row)["strict_event_candidate"] is False


def test_route_cannot_override_missing_atomic_evidence():
    row = clean()
    row["occurred_event_grounded"] = "uncertain"
    with pytest.raises(ValueError, match="strict route"):
        validate_result(row)
