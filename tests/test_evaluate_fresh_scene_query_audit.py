import pytest

from scripts.evaluate_fresh_scene_query_audit import evaluate


def _sealed(index, uid, modality, query_category="qcat", query="query"):
    return {
        "audit_index": index,
        "uid": uid,
        "title": uid,
        "source_modality": modality,
        "query_category": query_category,
        "query": query,
    }


def _blind(index, decision):
    return {"audit_index": index, "any_pillar_visual_candidate": decision}


def _post(index, strict, route, match="yes"):
    return {
        "audit_index": index,
        "assigned_label_match": match,
        "strict_pillar_pass": strict,
        "usable_route": route,
    }


def test_evaluate_reports_unit_and_source_deduplicated_yield():
    sealed = [
        _sealed(0, "same", "commentary"),
        _sealed(1, "same", "commentary"),
        _sealed(2, "other", "instructional"),
    ]
    blind = [_blind(0, "yes"), _blind(1, "yes"), _blind(2, "no")]
    post = [
        _post(0, True, "strict_commentary"),
        _post(1, True, "strict_commentary"),
        _post(2, False, "reject_visual", match="no"),
    ]
    report = evaluate(sealed, blind, post)
    assert report["overall"]["units"] == 3
    assert report["overall"]["sources"] == 2
    assert report["overall"]["strict_pass_units"] == 2
    assert report["overall"]["strict_pass_sources"] == 1
    assert report["overall"]["strict_pass_unit_rate"] == 2 / 3
    assert report["overall"]["strict_pass_source_rate"] == 1 / 2
    assert report["strict_source_contributions"][0]["strict_pass_units"] == 2
    assert report["decision"]["automatic_promotion"] is False


def test_evaluate_rejects_coverage_mismatch():
    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate(
            [_sealed(0, "one", "commentary")],
            [_blind(0, "yes")],
            [],
        )


def test_evaluate_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="duplicate audit_index"):
        evaluate(
            [_sealed(0, "one", "commentary"), _sealed(0, "two", "witnessed")],
            [_blind(0, "yes")],
            [_post(0, True, "strict_commentary")],
        )
