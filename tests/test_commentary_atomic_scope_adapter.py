import pytest

from scripts.commentary_occurred_event_contract_v3 import (
    validate_result,
    validate_result_atomic,
)


def atomic_clean():
    return {
        "actor_kind": "person",
        "behavior_kind": "physical_action",
        "affected_context_kind": "shared_social_context",
        "expectation_kind": "shared_coordination",
        "occurred_event_grounded": "yes",
        "social_actor_grounded": "yes",
        "behavior_semantically_specific": "yes",
        "target_or_shared_context_grounded": "yes",
        "normative_stance_grounded": "yes",
        "event_scope": "bounded_occurrence",
        "stance_quality": "unambiguous_external",
        "behavior_evidence_quote": "the driver entered the bike lane",
        "stance_evidence_quote": "that is dangerous and rude",
        "normalized_behavior": "a driver enters a bicycle lane",
        "normalized_norm": "keep vehicles out of bicycle lanes",
        "route": "strict_visual_search",
        "description": "One occurred maneuver and criticism are grounded.",
    }


def test_is_social_norm_is_derived_not_supplied():
    row = atomic_clean()
    assert "is_social_norm" not in row
    result = validate_result_atomic(row)
    assert result["is_social_norm"] == "yes"
    assert result["strict_event_candidate"] is True
    assert result["automatic_acceptance"] is False


def test_out_of_scope_atoms_block_strict_route():
    row = atomic_clean()
    row["expectation_kind"] = "technical_correctness"
    # Derived is_social_norm becomes "no", so the strict route must fail.
    with pytest.raises(ValueError, match="strict route"):
        validate_result_atomic(row)


def test_uncertain_atoms_fail_closed():
    row = atomic_clean()
    row["actor_kind"] = "uncertain"
    with pytest.raises(ValueError, match="strict route"):
        validate_result_atomic(row)


def test_contradictory_supplied_composite_is_rejected():
    row = atomic_clean()
    row["expectation_kind"] = "technical_correctness"
    row["is_social_norm"] = "yes"
    with pytest.raises(ValueError, match="contradicts atomic derivation"):
        validate_result_atomic(row)


def test_missing_atomic_fields_are_reported():
    row = atomic_clean()
    del row["behavior_kind"]
    with pytest.raises(ValueError, match="missing atomic scope"):
        validate_result_atomic(row)


def test_legacy_contract_still_accepts_direct_composite():
    # Backward compatibility: existing ledgers validated with the legacy
    # entry point continue to validate unchanged.
    row = {**atomic_clean(), "is_social_norm": "yes"}
    for field in ("actor_kind", "behavior_kind", "affected_context_kind", "expectation_kind"):
        del row[field]
    assert validate_result(row)["strict_event_candidate"] is True
