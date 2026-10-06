from __future__ import annotations

import pytest

from scripts.weak_label_contract import (
    attach_scope_labels,
    derive_concrete_behavior,
    derive_is_social_norm,
)


def scope(**overrides):
    values = {
        "actor_kind": "person",
        "behavior_kind": "physical_action",
        "affected_context_kind": "person",
        "expectation_kind": "interpersonal_treatment",
    }
    values.update(overrides)
    return derive_is_social_norm(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"actor_kind": "nonhuman"},
        {"behavior_kind": "technical_procedure"},
        {"behavior_kind": "accident_or_involuntary_event"},
        {"behavior_kind": "abstract_topic_or_trait"},
        {"affected_context_kind": "private_self_only"},
        {"affected_context_kind": "nonhuman_or_object_only"},
        {"expectation_kind": "technical_correctness"},
        {"expectation_kind": "legal_rule_only"},
        {"expectation_kind": "health_or_physical_safety_only"},
        {"expectation_kind": "personal_preference"},
        {"expectation_kind": "none"},
    ],
)
def test_explicit_exclusions_derive_no(overrides):
    assert scope(**overrides) == "no"


@pytest.mark.parametrize(
    "overrides",
    [
        {"actor_kind": "none"},
        {"actor_kind": "uncertain"},
        {"behavior_kind": "none"},
        {"behavior_kind": "uncertain"},
        {"affected_context_kind": "none"},
        {"affected_context_kind": "uncertain"},
        {"expectation_kind": "uncertain"},
    ],
)
def test_missing_or_unresolved_evidence_derives_uncertain(overrides):
    assert scope(**overrides) == "uncertain"


@pytest.mark.parametrize(
    "actor_kind,behavior_kind,affected_context_kind,expectation_kind",
    [
        ("person", "speech_act", "person", "interpersonal_treatment"),
        ("human_group", "omission", "shared_social_context", "shared_coordination"),
        (
            "institution",
            "institutional_action",
            "human_group",
            "civic_or_institutional_duty",
        ),
        (
            "person",
            "physical_action",
            "shared_social_context",
            "conventional_role_obligation",
        ),
    ],
)
def test_fully_grounded_in_scope_combinations_derive_yes(
    actor_kind, behavior_kind, affected_context_kind, expectation_kind
):
    assert (
        derive_is_social_norm(
            actor_kind=actor_kind,
            behavior_kind=behavior_kind,
            affected_context_kind=affected_context_kind,
            expectation_kind=expectation_kind,
        )
        == "yes"
    )


def test_concreteness_does_not_itself_imply_social_scope():
    assert derive_concrete_behavior("technical_procedure") == "yes"
    assert scope(behavior_kind="technical_procedure") == "no"


def test_attach_overwrites_any_free_form_composite_answer():
    result = {
        "actor_kind": "nonhuman",
        "behavior_kind": "physical_action",
        "affected_context_kind": "person",
        "expectation_kind": "interpersonal_treatment",
        "is_social_norm": "yes",
    }
    attached = attach_scope_labels(result)
    assert attached["is_social_norm"] == "no"
    assert attached["social_actor_grounded"] == "no"


def test_invalid_atomic_value_fails_closed():
    with pytest.raises(ValueError, match="invalid actor_kind"):
        scope(actor_kind="maybe-human")
