from scripts.evaluate_instructional_demo_consensus_v2 import (
    evaluate_rows,
    metric,
    v1_relaxed,
)


def v1(value: bool) -> dict:
    return {
        "visual_context": "connected_episode" if value else "presenter_or_interview",
        "participant_grounding": "affected_party_present" if value else "none",
        "physical_social_action": "yes" if value else "no",
        "quote_function": "none_or_unavailable",
    }


def v2(value: bool) -> dict:
    return {"result": {"demo_candidate": value}}


def test_relaxed_v1_ignores_completeness_but_requires_grounded_behavior() -> None:
    assert v1_relaxed(v1(True)) is True
    assert v1_relaxed(v1(False)) is False


def test_consensus_and_metrics_are_materialized() -> None:
    gold = {"a": (True, "demo"), "b": (False, "not demo")}
    rows, summary = evaluate_rows(
        "test", gold, {"a": v1(True), "b": v1(True)},
        {"a": v2(True), "b": v2(False)},
    )
    assert [row["v1_v2_consensus"] for row in rows] == [True, False]
    assert all(row["manual_output_reviewed"] for row in rows)
    assert summary["v1_v2_consensus"]["precision"] == 1.0
    assert summary["v1_v2_consensus"]["recall"] == 1.0


def test_metric_rejects_mismatched_or_empty_coverage() -> None:
    try:
        metric([], [])
    except ValueError as exc:
        assert "equal non-empty" in str(exc)
    else:
        raise AssertionError("empty coverage should fail")
