import pytest

from scripts.evaluate_fresh_scene_search_shadow import evaluate


def record(ordinal: int, *, query: str = "q", title: str = "t") -> dict:
    return {
        "ordinal": ordinal,
        "uid": f"u{ordinal}",
        "artifact_status": "rendered",
        "group": "g",
        "query": query,
        "title": title,
    }


def review(ordinal: int, *, strict: str = "no") -> dict:
    return {
        "ordinal": ordinal,
        "uid": f"u{ordinal}",
        "artifact_status": "rendered",
        "group": "g",
        "visual_scene_candidate": "yes",
        "target_query_match": "yes",
        "strict_pillar_candidate": strict,
    }


def test_evaluate_reports_bounds_and_title_deduplication():
    report = evaluate(
        {"records": [record(0), record(1), record(2, title="other")]},
        [review(0, strict="yes"), review(1), review(2, strict="uncertain")],
    )
    assert report["overall"]["strict_lower_bound_rate"] == pytest.approx(1 / 3)
    assert report["overall"]["strict_upper_bound_rate"] == pytest.approx(2 / 3)
    assert report["unique_query_title_overall"]["rendered"] == 2


def test_evaluate_refuses_partial_review():
    with pytest.raises(ValueError, match="incomplete review"):
        evaluate({"records": [record(0), record(1)]}, [review(0)])
