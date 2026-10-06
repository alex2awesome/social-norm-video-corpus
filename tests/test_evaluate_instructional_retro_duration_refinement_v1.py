from scripts.evaluate_instructional_retro_duration_refinement_v1 import evaluate, rate


def contract():
    return {
        "status": "discovery_only",
        "signal": {"query_source": "retro_instr_scan", "minimum_duration_seconds": 5},
        "cohorts": [{"name": "a"}, {"name": "b"}],
    }


def row(cohort, source, duration, visual, usable):
    return {
        "cohort": cohort, "query_source": source, "duration_seconds": duration,
        "manual_visual_demo": visual, "manual_usable_demo": usable,
    }


def test_refinement_reports_each_frozen_cohort_and_never_promotes():
    rows = [
        row("a", "retro_instr_scan", 4, False, False),
        row("a", "retro_instr_scan", 6, True, True),
        row("b", "retro_instr_scan", 7, True, False),
        row("b", "other", 9, False, False),
    ]
    report = evaluate(rows, contract())
    assert report["cohorts"]["a"]["manual_visual_demo"]["retro_only"]["rate"] == 0.5
    assert report["cohorts"]["a"]["manual_visual_demo"]["retro_and_min_duration"]["rate"] == 1.0
    assert report["aggregate"]["manual_visual_demo"]["retro_and_min_duration"]["selected"] == 2
    assert report["future_replication_required"] is True
    assert report["operational_rule_promoted"] is False
    assert report["automatic_acceptance"] is False


def test_empty_selection_has_no_invented_precision():
    metrics = rate(
        [row("a", "other", 2, False, False)],
        lambda value: value["query_source"] == "retro_instr_scan",
        "manual_visual_demo",
    )
    assert metrics == {"selected": 0, "positive": 0, "rate": None}
