import pytest

from scripts.witnessed_reaction_av_contract import candidate_positive


def positive():
    return {
        "reaction_grounded": "yes",
        "action_before_or_overlaps_response": "yes",
        "response_targets_action": "yes",
        "responder_role": "separate_bystander",
        "response_content": "targeted_objection",
        "trigger_kind": "interpersonal_treatment",
        "staging": "no_clear_staging_evidence",
    }


def test_strict_contract_accepts_only_complete_atomic_conjunction():
    assert candidate_positive(
        positive(), include_authority=False, require_social=True
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reaction_grounded", "uncertain"),
        ("action_before_or_overlaps_response", "no"),
        ("response_targets_action", "uncertain"),
        ("responder_role", "actor_or_target"),
        ("response_content", "generic_surprise_or_affect"),
        ("trigger_kind", "private_physical_safety"),
        ("staging", "clearly_staged"),
        ("staging", "uncertain"),
    ],
)
def test_strict_social_contract_fails_closed_on_each_required_atom(field, value):
    result = positive()
    result[field] = value
    assert not candidate_positive(
        result, include_authority=False, require_social=True
    )


def test_authority_and_social_scope_are_independent_switches():
    result = positive()
    result["responder_role"] = "authority_or_host"
    result["trigger_kind"] = "private_physical_safety"
    assert not candidate_positive(
        result, include_authority=False, require_social=False
    )
    assert candidate_positive(
        result, include_authority=True, require_social=False
    )
    assert not candidate_positive(
        result, include_authority=True, require_social=True
    )


def test_missing_staging_fails_closed():
    result = positive()
    del result["staging"]
    assert not candidate_positive(
        result, include_authority=False, require_social=True
    )
