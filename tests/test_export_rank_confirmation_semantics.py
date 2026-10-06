import pytest

from scripts.export_rank_confirmation_semantics import aligned_segments, select_rows


def test_select_rows_preserves_frozen_audit_order():
    corpus = [
        {"item_id": "a", "uid": "ua", "norm": "x"},
        {"item_id": "b", "uid": "ub", "norm": "y"},
    ]
    selection = [
        {"audit_index": 1, "item_id": "a", "uid": "ua"},
        {"audit_index": 0, "item_id": "b", "uid": "ub"},
    ]
    result = select_rows(corpus, selection)
    assert [row["item_id"] for row in result] == ["b", "a"]
    assert [row["audit_index"] for row in result] == [0, 1]


def test_select_rows_rejects_uid_mismatch():
    with pytest.raises(ValueError, match="uid mismatch"):
        select_rows(
            [{"item_id": "a", "uid": "ua"}],
            [{"audit_index": 0, "item_id": "a", "uid": "wrong"}],
        )


def test_aligned_segments_uses_overlap_and_tolerance():
    transcript = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "before"},
            {"start": 1.8, "end": 2.2, "text": "edge"},
            {"start": 3.0, "end": 4.0, "text": "inside"},
            {"start": 5.6, "end": 6.0, "text": "after"},
        ]
    }
    result = aligned_segments(transcript, 2.5, 5.0)
    assert [row["text"] for row in result] == ["edge", "inside"]
