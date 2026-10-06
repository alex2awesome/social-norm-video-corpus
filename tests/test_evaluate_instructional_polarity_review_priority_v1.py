import pytest

from scripts.evaluate_instructional_polarity_review_priority_v1 import (
    evaluate,
    fresh_rows,
    metric,
)


def row(polarity: str, visual: bool) -> dict:
    return {"uid": f"{polarity}-{visual}", "polarity": polarity, "visual_demo": visual}


def test_metric_preserves_explanation_false_negatives_and_reports_recall():
    rows = [row("violation", True), row("correct", False), row("explanation", True)]
    result = metric(rows)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["recall"] == 0.5
    assert result["explanation_visual_demo_rate"] == 1.0


def test_review_ranking_gate_can_pass_without_accept_or_reject():
    cohort = [
        {"uid": f"n-{i}", "polarity": "violation", "visual_demo": i < 18}
        for i in range(73)
    ] + [
        {"uid": f"e-{i}", "polarity": "explanation", "visual_demo": i < 2}
        for i in range(27)
    ]
    report = evaluate({"one": cohort, "two": cohort})
    assert report["review_ranking_promoted"] is True
    assert report["allowed_use"] == "manual_review_ranking_only"
    assert report["automatic_acceptance"] is False
    assert report["automatic_rejection"] is False
    assert report["explanation_policy"].endswith("never_reject")


def test_fresh_rows_requires_exact_manual_coverage():
    with pytest.raises(ValueError, match="exactly cover"):
        fresh_rows(
            [{"audit_index": 0, "uid": "a", "polarity": "violation"}],
            [],
        )
