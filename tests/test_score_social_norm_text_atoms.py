import json

import pytest

from scripts.score_social_norm_text_atoms import (
    derive_social_candidate,
    evidence_is_grounded,
    parse_result,
)


def result(**overrides):
    value = {
        "actor_scope": "interpersonal_actors",
        "behavior_scope": "interpersonal_conduct",
        "expectation_scope": "interpersonal_treatment",
        "concrete_behavior_grounded": "yes",
        "affected_person_or_shared_context_grounded": "yes",
        "assigned_norm_relation": "exact",
        "normalized_behavior": "one person insults another",
        "evidence": "insults another",
        "evidence_in_aligned_transcript": True,
        "reason": "A concrete interpersonal behavior is grounded.",
    }
    value.update(overrides)
    return value


def test_candidate_is_derived_from_all_atomic_fields():
    assert derive_social_candidate(result())
    for field, value in (
        ("actor_scope", "private_self"),
        ("behavior_scope", "technical_or_procedural"),
        ("expectation_scope", "identity_belief_or_ritual"),
        ("concrete_behavior_grounded", "no"),
        ("affected_person_or_shared_context_grounded", "no"),
        ("assigned_norm_relation", "unsupported"),
        ("evidence_in_aligned_transcript", False),
    ):
        assert not derive_social_candidate(result(**{field: value}))


def test_parser_rejects_unknown_enum():
    with pytest.raises(ValueError, match="invalid behavior_scope"):
        parse_result(json.dumps(result(behavior_scope="vague_goodness")))


def test_evidence_grounding_is_normalized_but_exact():
    assert evidence_is_grounded("insults   another", "He INSULTS another person.")
    assert not evidence_is_grounded("insults the other person", "He insults another person.")
    assert not evidence_is_grounded("none", "none of that happened")
