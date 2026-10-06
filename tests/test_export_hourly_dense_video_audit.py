import pytest

from scripts.export_hourly_dense_video_audit import select_rows


def test_select_rows_requires_full_identity_matched_review():
    source = {"audit_index": 0, "uid": "x"}
    review = {
        "audit_index": 0,
        "uid": "x",
        "dense_followup": True,
        "evidence": "manual",
    }

    assert select_rows({"visual_samples": [source]}, [review]) == [(source, review)]

    with pytest.raises(ValueError, match="complete source manifest"):
        select_rows({"visual_samples": [source]}, [])


def test_select_rows_excludes_manual_non_candidates():
    source = {"audit_index": 0, "uid": "x"}
    review = {
        "audit_index": 0,
        "uid": "x",
        "dense_followup": False,
        "evidence": "manual",
    }

    assert select_rows({"visual_samples": [source]}, [review]) == []
