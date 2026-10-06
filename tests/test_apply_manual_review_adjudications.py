import pytest

from scripts.apply_manual_review_adjudications import apply_adjudications


def test_applies_full_replacement_without_reordering():
    base = [
        {"item_id": "a", "decision": "no"},
        {"item_id": "b", "decision": "yes"},
    ]
    output = apply_adjudications(
        base,
        [
            {
                "item_id": "a",
                "reason": "dense source evidence",
                "replacement": {"item_id": "a", "decision": "yes"},
            }
        ],
    )
    assert output == [
        {"item_id": "a", "decision": "yes"},
        {"item_id": "b", "decision": "yes"},
    ]
    assert base[0]["decision"] == "no"


def test_requires_exact_replacement_schema():
    with pytest.raises(ValueError, match="schema mismatch"):
        apply_adjudications(
            [{"item_id": "a", "decision": "no"}],
            [
                {
                    "item_id": "a",
                    "reason": "new evidence",
                    "replacement": {"item_id": "a", "decision": "yes", "extra": 1},
                }
            ],
        )
