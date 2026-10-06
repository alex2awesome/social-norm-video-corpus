import pytest

from scripts.materialize_commentary_tail_trim_experiment import tail_bounds, unique_eligible_parents


def test_tail_bounds_are_fixed_last_half() -> None:
    assert tail_bounds(6.0) == (3.0, 6.0)
    assert tail_bounds(7.5) == (3.75, 7.5)
    with pytest.raises(ValueError, match="too short"):
        tail_bounds(1.5)


def test_unique_parents_collapses_spatial_variants(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.materialize_commentary_tail_trim_experiment.derive_trials",
        lambda *_args: [
            {"parent_candidate_id": "a", "parent_audit_index": 2, "variant": "x"},
            {"parent_candidate_id": "a", "parent_audit_index": 2, "variant": "y"},
            {"parent_candidate_id": "b", "parent_audit_index": 5, "variant": "x"},
            {"parent_candidate_id": "b", "parent_audit_index": 5, "variant": "y"},
        ],
    )
    result = unique_eligible_parents([], [], [])
    assert [row["parent_candidate_id"] for row in result] == ["a", "b"]
