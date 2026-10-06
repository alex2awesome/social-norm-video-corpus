import pytest

from scripts.filter_witnessed_scores_to_selection import filter_scores


def selection():
    return [{
        "candidates": [{"candidate_id": "b"}, {"candidate_id": "a"}],
    }]


def test_filters_latest_successes_in_selection_order():
    rows = filter_scores(selection(), [
        {"candidate_id": "a", "error": "old failure"},
        {"candidate_id": "a", "error": None, "result": {"value": "a"}},
        {"candidate_id": "b", "error": None, "result": {"value": "b"}},
        {"candidate_id": "outside", "error": None, "result": {}},
    ])
    assert [row["candidate_id"] for row in rows] == ["b", "a"]


def test_rejects_missing_success():
    with pytest.raises(ValueError, match="missing 1"):
        filter_scores(selection(), [
            {"candidate_id": "a", "error": None, "result": {}},
            {"candidate_id": "b", "error": "failed", "result": None},
        ])
