from __future__ import annotations

import pytest

from scripts.summarize_instructional_manual_contract_routes import (
    manual_route,
    summarize,
)


def row(item, uid, *, visual="yes", aligned="yes", complete="yes", usable="yes"):
    return {
        "item_id": item,
        "uid": uid,
        "visual_form": "animation",
        "visual_demo": visual,
        "semantic_alignment": aligned,
        "complete_demo": complete,
        "usable_demo": usable,
    }


def test_manual_routes_preserve_relabel_and_recut_candidates():
    assert manual_route(row("a", "u-a")) == "instructional_demo_candidate"
    assert manual_route(row(
        "b", "u-b", aligned="no", usable="no"
    )) == "instructional_demo_candidate_after_relabel"
    assert manual_route(row(
        "c", "u-c", complete="no", usable="no"
    )) == "instructional_source_needs_recut"
    assert manual_route(row(
        "d", "u-d", visual="no", complete="no", usable="no"
    )) == "text_only_instructional"


def test_summary_keeps_cohorts_separate_and_reports_overlap():
    report = summarize({
        "a": [row("a", "shared")],
        "b": [row("b", "shared", aligned="no", usable="no")],
    })
    assert report["items"] == 2
    assert report["unique_sources"] == 1
    assert report["cross_cohort_overlapping_sources"] == 1
    assert report["cohort_rates_must_be_reported_separately"] is True
    assert report["automatic_acceptance"] is False


def test_usable_demo_must_agree_with_atomic_manual_labels():
    with pytest.raises(ValueError, match="contradicts atoms"):
        manual_route(row("a", "u-a", visual="no"))


def test_each_cohort_must_be_source_disjoint():
    with pytest.raises(ValueError, match="source-disjoint"):
        summarize({"a": [row("a", "same"), row("b", "same")]})

