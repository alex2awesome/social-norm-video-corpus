import json

from weaksup.fetch_reddit_context_v1 import compact_comment, compact_post
from weaksup.harvest_info_sidecars_v1 import compact_record


def test_sidecar_compaction_keeps_incident_fields_and_truncates():
    record = compact_record("dailymotion__x1", {
        "title": "Man cuts entire queue",
        "description": "Occurred on August 2022. " + "x" * 5000,
        "tags": ["queue", "rude"], "view_count": 1234,
        "irrelevant_huge_field": {"formats": ["..."] * 100},
        "uploader": "ViralHog", "language": None, "categories": [],
    })
    assert record["source"] == "dailymotion"
    assert record["title"] == "Man cuts entire queue"
    assert len(record["description"]) == 4000
    assert record["view_count"] == 1234
    assert "irrelevant_huge_field" not in record
    assert "language" not in record and "categories" not in record


def test_reddit_compaction():
    post = compact_post({"id": "1uavwg8", "title": "Creep waits...",
                         "subreddit": "BlatantMisogyny", "score": 900,
                         "selftext": "", "author": "someone", "over_18": False})
    assert post["subreddit"] == "BlatantMisogyny"
    assert "selftext" not in post
    assert post["over_18"] is False
    comment = compact_comment({"body": "b" * 2000, "score": 5, "id": "c1"})
    assert len(comment["body"]) == 1500 and comment["score"] == 5
