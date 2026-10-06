import pytest

from scripts.materialize_title_event_shadow_clips import select_recovered


def test_select_recovered_requires_strict_and_preserves_audit_order():
    candidates = [
        {"item_id": "a", "source_path": "/tmp/a.mp4", "uid": "ua"},
        {"item_id": "b", "source_path": "/tmp/b.mp4", "uid": "ub"},
    ]
    dense = [
        {
            "audit_index": 2,
            "item_id": "b",
            "dense_outcome": "strict_recovered",
            "event_start_sec": 2,
            "event_end_sec": 3,
        },
        {
            "audit_index": 0,
            "item_id": "a",
            "dense_outcome": "still_unresolved",
            "event_start_sec": 0,
            "event_end_sec": 1,
        },
    ]
    selected = select_recovered(candidates, dense)
    assert [row[1]["item_id"] for row in selected] == ["b"]


def test_select_recovered_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="invalid event bounds"):
        select_recovered(
            [{"item_id": "a", "source_path": "/tmp/a.mp4", "uid": "ua"}],
            [{
                "audit_index": 0,
                "item_id": "a",
                "dense_outcome": "strict_recovered",
                "event_start_sec": 2,
                "event_end_sec": 1,
            }],
        )
