import pytest

from scripts.evaluate_commentary_title_event_cues import evaluate, wilson


def test_wilson_is_bounded():
    lower, upper = wilson(5, 10)
    assert 0 < lower < 0.5 < upper < 1


def test_evaluate_requires_complete_identity_and_reports_yield():
    blind = [{"audit_index": 0, "item_id": "i", "uid": "u"}]
    review = [{"audit_index": 0, "social_event_visible": "yes"}]
    sealed = [{"audit_index": 0, "item_id": "i", "uid": "u", "score_band": "b"}]
    post = [
        {
            "audit_index": 0,
            "disposition": "title_labeled_visual_event",
        }
    ]
    candidates = [{"item_id": "i", "title_event_cues": ["actor_action"]}]
    report = evaluate(blind, review, sealed, post, candidates, 100)
    assert report["overall"]["strict_rate"] == 1
    assert report["estimated_strict_survivors"] == 100

    with pytest.raises(ValueError, match="identity mismatch"):
        evaluate(
            blind,
            review,
            [{**sealed[0], "uid": "wrong"}],
            post,
            candidates,
            100,
        )
