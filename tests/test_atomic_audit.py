from __future__ import annotations

import pytest

from scripts.run_openai_atomic_audit import atomic_schema, derive_and_validate


@pytest.mark.parametrize("pillar", ["instructional", "witnessed", "commentary"])
def test_model_schema_asks_atomic_scope_not_composite_scope(pillar):
    schema = atomic_schema(pillar)
    assert "is_social_norm" not in schema["properties"]
    for field in (
        "actor_kind",
        "behavior_kind",
        "affected_context_kind",
        "expectation_kind",
    ):
        assert field in schema["properties"]
        assert field in schema["required"]


def commentary_result(**overrides):
    result = {
        "actor_kind": "person",
        "behavior_kind": "speech_act",
        "affected_context_kind": "person",
        "expectation_kind": "interpersonal_treatment",
        "normative_stance_grounded": "yes",
        "proposed_norm_supported": "yes",
        "stance_type": "criticism",
        "quote_relation": "same_sentence",
        "decision": "accept",
        "rejection_reasons": [],
        "normalized_behavior": "A insults B.",
        "normalized_norm": "People should not insult others.",
        "behavior_evidence_quote": "A insults B",
        "stance_evidence_quote": "A insults B",
        "description": "Grounded insult and criticism.",
    }
    result.update(overrides)
    return result


def commentary_item():
    return {
        "start_quote": "A insults B and that is wrong",
        "transcript_context": [],
    }


def test_composite_scope_is_attached_before_existing_pillar_validation():
    result = derive_and_validate("commentary", commentary_result(), commentary_item())
    assert result["is_social_norm"] == "yes"
    assert result["social_actor_grounded"] == "yes"
    assert result["concrete_behavior"] == "yes"
    assert result["target_or_shared_context_grounded"] == "yes"


def test_out_of_scope_atomic_answer_cannot_be_rescued_by_accept_decision():
    with pytest.raises(ValueError, match="grounding invariants"):
        derive_and_validate(
            "commentary",
            commentary_result(behavior_kind="technical_procedure"),
            commentary_item(),
        )
