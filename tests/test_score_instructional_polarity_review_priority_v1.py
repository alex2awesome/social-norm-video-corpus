import json

from scripts.score_instructional_polarity_review_priority_v1 import (
    score_corpus,
    score_demo,
)


def test_non_explanation_is_ranked_but_never_accepted():
    row = score_demo("u", 0, {"polarity": "Violation", "clip": "demo_0.mp4"})
    assert row["instructional_non_explanation_review_priority_v1"] == "violation"
    assert row["priority_tier"] == "standard_above_explanation"
    assert row["automatic_acceptance"] is False
    assert row["automatic_rejection"] is False
    assert row["delete_media"] is False


def test_explanation_is_preserved_at_lower_priority():
    row = score_demo("u", 1, {"polarity": "explanation"})
    assert row["instructional_non_explanation_review_priority_v1"] is None
    assert row["priority_tier"] == "lower_priority_preserved"
    assert row["review_route"] == "instructional_demo_manual_review"
    assert row["automatic_rejection"] is False


def test_corpus_scorer_emits_every_demo_and_preserves_bad_sources(tmp_path):
    first = tmp_path / "data" / "instructional" / "one"
    first.mkdir(parents=True)
    (first / "metadata.json").write_text(json.dumps({
        "demos": [
            {"polarity": "correct", "clip": "a.mp4"},
            {"polarity": "explanation", "clip": "b.mp4"},
        ]
    }))
    second = tmp_path / "data" / "instructional" / "two"
    second.mkdir(parents=True)
    (second / "metadata.json").write_text("not json")
    rows = score_corpus(tmp_path)
    assert len(rows) == 3
    assert [row["uid"] for row in rows] == ["one", "one", "two"]
    assert rows[2]["priority_tier"] == "unscored_preserved"
    assert rows[2]["automatic_rejection"] is False
