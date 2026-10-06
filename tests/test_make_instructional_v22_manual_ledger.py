import pytest

from scripts.make_instructional_v22_manual_ledger import template_rows


def test_template_rows_are_complete_and_blind() -> None:
    rows = template_rows(
        [
            {"audit_index": 1, "candidate_id": "c1", "item_id": "i1", "uid": "u1"},
            {"audit_index": 0, "candidate_id": "c0", "item_id": "i0", "uid": "u0"},
        ]
    )
    assert [row["item_id"] for row in rows] == ["i0", "i1"]
    assert rows[0]["visual_demo"] == ""
    assert "norm" not in rows[0]


def test_template_rejects_duplicate_items() -> None:
    manifest = [
        {"audit_index": 0, "candidate_id": "c0", "item_id": "same", "uid": "u0"},
        {"audit_index": 1, "candidate_id": "c1", "item_id": "same", "uid": "u1"},
    ]
    with pytest.raises(ValueError, match="duplicate item_id"):
        template_rows(manifest)
