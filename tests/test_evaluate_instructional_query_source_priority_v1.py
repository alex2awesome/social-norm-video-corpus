import pytest

from scripts.evaluate_instructional_query_source_priority_v1 import evaluate


def row(cohort: str, uid: str, selected: bool, visual: bool) -> dict:
    return {
        "cohort": cohort,
        "uid": uid,
        "retro_priority": selected,
        "manual_visual_demo": visual,
        "manual_usable_demo": visual,
    }


def contract() -> dict:
    return {
        "status": "retrospective",
        "policy": "review_only",
        "cohorts": [{"name": "a"}, {"name": "b"}],
        "gate": {
            "source_uid_overlap": 0,
            "minimum_total_selected": 2,
            "minimum_selected_visual_precision_each_cohort": 0.5,
            "minimum_visual_precision_delta_each_cohort": 0.3,
        },
    }


def test_replication_gate_only_authorizes_review_priority() -> None:
    rows = [
        row("a", "a1", True, True), row("a", "a2", False, False),
        row("b", "b1", True, True), row("b", "b2", False, False),
    ]
    report = evaluate(rows, contract())
    assert report["review_ranking_gate_passed"] is True
    assert report["allowed_use"] == "manual_review_priority_only"
    assert report["automatic_acceptance"] is False
    assert report["corpus_mutated"] is False


def test_one_cohort_failure_disables_signal() -> None:
    rows = [
        row("a", "a1", True, True), row("a", "a2", False, False),
        row("b", "b1", True, False), row("b", "b2", False, True),
    ]
    report = evaluate(rows, contract())
    assert report["review_ranking_gate_passed"] is False
    assert report["allowed_use"] == "untrusted_diagnostic_only"


def test_source_overlap_fails_closed() -> None:
    rows = [
        row("a", "same", True, True), row("a", "a2", False, False),
        row("b", "same", True, True), row("b", "b2", False, False),
    ]
    report = evaluate(rows, contract())
    assert report["gate_checks"]["source_uid_overlap"] is False
    assert report["review_ranking_gate_passed"] is False
