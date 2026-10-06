import pytest

from scripts.evaluate_instructional_retro_duration_holdout_v1 import evaluate


def item(candidate, cohort, uid):
    return {"candidate_id": candidate, "holdout_cohort": cohort, "uid": uid}


def blind(candidate, visual="yes"):
    return {
        "candidate_id": candidate, "visual_demo": visual,
        "complete_demo": "yes" if visual == "yes" else "no",
        "visual_form": "live_scene", "participant_grounding": "actor_and_recipient",
        "blind_evidence": "Two people perform a connected interaction.",
    }


def post(candidate, usable="yes", exact="yes", concrete="yes", corrected=""):
    return {
        "candidate_id": candidate, "label_is_concrete_social_behavior": concrete,
        "visual_matches_original_label": exact, "usable_after_relabel": usable,
        "visible_polarity": "violation", "failure_mode": "none",
        "corrected_behavior": corrected,
        "post_reveal_evidence": "The visible interaction supports the proposed behavior.",
    }


def prereg(items=3, minimum=1):
    return {
        "items": items,
        "gate": {
            "minimum_signal_positive_items": minimum,
            "minimum_visual_demo_precision": 0.7,
            "minimum_visual_demo_wilson_95_lower": 0.0,
        },
    }


def test_reports_cohorts_separately_and_only_promotes_review_priority():
    sealed = [item("a", "signal_positive", "u1"), item("b", "short_retro_boundary", "u2"), item("c", "duration_matched_nonretro_control", "u3")]
    report, rows = evaluate(
        sealed, [blind("a"), blind("b", "no"), blind("c", "no")],
        [post("a"), post("b", "no"), post("c", "no")], prereg(),
    )
    assert report["cohorts"]["signal_positive"]["visual_demo_precision"] == 1.0
    assert report["review_priority_rule_promoted"] is True
    assert report["allowed_use"] == "manual_review_priority_only"
    assert report["automatic_acceptance"] is False
    assert len(rows) == 3


def test_non_demo_cannot_be_marked_usable():
    with pytest.raises(ValueError, match="non-demo"):
        evaluate(
            [item("a", "signal_positive", "u1")], [blind("a", "no")],
            [post("a", "yes")], prereg(items=1),
        )


def test_incomplete_manual_coverage_fails_closed():
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate(
            [item("a", "signal_positive", "u1")], [blind("a")], [], prereg(items=1)
        )


def test_duplicate_source_is_rejected():
    with pytest.raises(ValueError, match="source-disjoint"):
        evaluate(
            [item("a", "signal_positive", "u1"), item("b", "short_retro_boundary", "u1")],
            [blind("a"), blind("b")], [post("a"), post("b")], prereg(items=2),
        )


def test_relabel_requires_an_explicit_corrected_behavior():
    with pytest.raises(ValueError, match="corrected_behavior"):
        evaluate(
            [item("a", "signal_positive", "u1")], [blind("a")],
            [post("a", usable="yes", exact="no")], prereg(items=1),
        )


def test_abstract_label_cannot_be_marked_exact():
    with pytest.raises(ValueError, match="abstract label"):
        evaluate(
            [item("a", "signal_positive", "u1")], [blind("a")],
            [post("a", exact="yes", concrete="no")], prereg(items=1),
        )
