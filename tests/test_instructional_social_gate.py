import json

import pytest

from scripts.score_instructional_social_gate import derive_band, parse_json


def result(**overrides):
    value = {
        "actor_kind": "person",
        "behavior_kind": "interpersonal_physical_action",
        "affected_context_kind": "visible_person",
        "expectation_kind": "interpersonal_treatment",
        "depiction_support": "complete",
        "grounded_quote_names_concrete_behavior": "yes",
        "grounded_quote_behavior_occurs_in_depiction": "yes",
        "audience_presentation_only": "no",
        "technical_or_nonsocial_activity_only": "no",
        "label_relation": "exact",
        "normalized_behavior": "person grabs another person",
        "failure_reason": "none",
        "reason": "The described grab occurs between visible people.",
    }
    value.update(overrides)
    return value


def test_exact_and_relabel_bands_are_derived_from_atoms():
    assert derive_band(result()) == "v7_exact_candidate"
    assert (
        derive_band(result(label_relation="wrong_but_relabelable"))
        == "v7_relabel_candidate"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("actor_kind", "nonhuman_or_object"),
        ("behavior_kind", "generic_motion_or_speech"),
        ("affected_context_kind", "implicit_or_offscreen_target"),
        ("expectation_kind", "health_or_safety_only"),
        ("grounded_quote_names_concrete_behavior", "no"),
        ("grounded_quote_behavior_occurs_in_depiction", "no"),
        ("audience_presentation_only", "yes"),
        ("technical_or_nonsocial_activity_only", "yes"),
    ],
)
def test_each_required_atom_can_block_candidate(field, value):
    assert derive_band(result(**{field: value})) == "v7_semantic_or_domain_review"


def test_partial_complete_atoms_route_to_recut_review():
    assert derive_band(result(depiction_support="partial")) == "v7_recut_review"


def test_parser_rejects_unknown_enum():
    row = result(actor_kind="robot")
    with pytest.raises(ValueError, match="invalid actor_kind"):
        parse_json(json.dumps(row))
