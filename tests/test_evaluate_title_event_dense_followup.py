import pytest

from scripts.evaluate_title_event_dense_followup import evaluate


def post(index, disposition):
    return {"audit_index": index, "disposition": disposition}


def dense(index, outcome, disposition):
    return {
        "audit_index": index,
        "dense_outcome": outcome,
        "final_disposition": disposition,
    }


def test_evaluate_replaces_only_followup_dispositions():
    report = evaluate(
        [
            post(0, "title_labeled_visual_event"),
            post(1, "dense_review"),
            post(2, "crop_review"),
        ],
        [
            dense(1, "strict_recovered", "title_labeled_visual_event"),
            dense(2, "not_usable", "event_absent"),
        ],
        30,
    )
    assert report["strict_usable"] == 2
    assert report["unresolved"] == 0
    assert report["dense_outcomes"] == {
        "strict_recovered": 1,
        "not_usable": 1,
    }


def test_evaluate_requires_exact_followup_coverage():
    with pytest.raises(ValueError, match="every and only"):
        evaluate(
            [post(0, "dense_review")],
            [],
            1,
        )
