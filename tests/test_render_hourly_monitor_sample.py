import pytest

from scripts.render_hourly_monitor_sample import select_queue


def test_select_queue_requires_absolute_unique_media_paths():
    rows = select_queue(
        {
            "fresh_review_queue": [
                {"uid": "a", "media_path": "/data/a.mp4"},
                {"uid": "b", "media_path": "/data/b.mp4"},
            ]
        }
    )
    assert [row["uid"] for row in rows] == ["a", "b"]

    with pytest.raises(ValueError, match="duplicate"):
        select_queue(
            {
                "fresh_review_queue": [
                    {"uid": "a", "media_path": "/data/a.mp4"},
                    {"uid": "a", "media_path": "/data/b.mp4"},
                ]
            }
        )

    with pytest.raises(ValueError, match="absolute"):
        select_queue(
            {"fresh_review_queue": [{"uid": "a", "media_path": "data/a.mp4"}]}
        )
