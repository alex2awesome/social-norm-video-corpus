import pytest

from scripts.validate_audited_search_queries_v1 import evaluate


def fixtures():
    config = {
        "policy": {"retrieval_only": True, "query_text_is_never_a_label": True},
        "audit": {
            "reviewed_sources": 2,
            "selected_query_reviewed_sources": 1,
            "selected_query_strict_target_passes": 1,
            "excluded_zero_pass_query_formulations": 1,
            "excluded_witnessed_query_formulations": 0,
        },
        "queries": [{
            "pillar": "instructional", "query": "acted scenario",
            "reviewed": 1, "strict_target_passes": 1,
        }],
    }
    selection = {"items": [
        {"uid": "u1", "pillar": "instructional", "query": "acted scenario"},
        {"uid": "u2", "pillar": "instructional", "query": "lecture"},
    ]}
    manual = [
        {
            "uid": "u1", "intended_pillar": "instructional",
            "strict_target_pass": True, "visual_scene": "yes",
            "label_support": "exact",
        },
        {
            "uid": "u2", "intended_pillar": "instructional",
            "strict_target_pass": False, "visual_scene": "no",
            "label_support": "text_only",
        },
    ]
    return config, selection, manual


def test_validates_exact_manual_query_counts_without_acceptance() -> None:
    config, selection, manual = fixtures()
    report = evaluate(config, selection, manual)
    assert report["configured_strict_target_passes"] == 1
    assert report["omitted_zero_yield_queries"] == 1
    assert report["automatic_acceptance"] is False
    assert report["corpus_mutated"] is False


def test_rejects_zero_yield_production_query() -> None:
    config, selection, manual = fixtures()
    config["queries"] = [{
        "pillar": "instructional", "query": "lecture",
        "reviewed": 1, "strict_target_passes": 0,
    }]
    with pytest.raises(ValueError, match="zero-yield"):
        evaluate(config, selection, manual)


def test_rejects_witnessed_query_even_with_one_apparent_pass() -> None:
    config, selection, manual = fixtures()
    config["queries"] = [{
        "pillar": "witnessed", "query": "confronted", "reviewed": 1,
        "strict_target_passes": 1,
    }]
    config["audit"].update({
        "selected_query_reviewed_sources": 1,
        "selected_query_strict_target_passes": 1,
    })
    selection["items"][0].update({"pillar": "witnessed", "query": "confronted"})
    manual[0]["intended_pillar"] = "witnessed"
    with pytest.raises(ValueError, match="failed its pillar contract"):
        evaluate(config, selection, manual)


def test_rejects_strict_pass_without_visual_support() -> None:
    config, selection, manual = fixtures()
    manual[0]["visual_scene"] = "no"
    with pytest.raises(ValueError, match="lacks visual label support"):
        evaluate(config, selection, manual)


def test_requires_exact_source_disjoint_manual_coverage() -> None:
    config, selection, manual = fixtures()
    manual.pop()
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate(config, selection, manual)
