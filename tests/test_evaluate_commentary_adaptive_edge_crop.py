import pytest

from scripts.evaluate_commentary_adaptive_edge_crop import evaluate


def manifest():
    return [{
        "candidate_id": "c", "parent_candidate_id": "p",
        "proxy_clip": "/c.mp4", "proxy_clip_sha256": "hash",
    }]


def blind(**overrides):
    row = {
        "candidate_id": "c", "performed_event_visible": "yes",
        "actor_target_grounded": "yes", "start_boundary_clean": "yes",
        "end_boundary_clean": "yes", "label_bearing_text_absent": "yes",
        "editorial_target_marker_absent": "yes", "audio_absent": "yes",
        "medium": "live_action", "blind_evidence": "A visible interaction occurs.",
    }
    row.update(overrides)
    return [row]


def post():
    return [{
        "candidate_id": "c", "exact_named_action_visible": "yes",
        "label_alignment": "exact", "route": "commentary_visual",
        "post_reveal_evidence": "The visible event exactly matches the title.",
    }]


def materialization(cohort=1, abstentions=None):
    abstentions = abstentions or []
    return {
        "cohort_items": cohort, "rendered_items": 1,
        "abstentions": abstentions,
    }


def contract(minimum_rendered=1):
    return {"gate": {
        "minimum_rendered_outputs": minimum_rendered,
        "minimum_usable_rate": 0.8,
        "minimum_parent_salvage_rate": 0.5,
    }}


def test_complete_clean_artifact_can_only_request_fresh_holdout() -> None:
    report, accepted = evaluate(
        manifest(), blind(), post(), materialization(), contract()
    )
    assert report["candidate_for_fresh_source_disjoint_holdout"] is True
    assert report["automatic_keep_rule_promoted"] is False
    assert len(accepted) == 1


def test_editorial_marker_fails_closed() -> None:
    report, accepted = evaluate(
        manifest(), blind(editorial_target_marker_absent="no"), post(),
        materialization(), contract(),
    )
    assert accepted == []
    assert report["failure_counts"]["editorial_target_marker_absent"] == 1


def test_abstentions_remain_in_cohort_denominator() -> None:
    report, _ = evaluate(
        manifest(), blind(), post(),
        materialization(2, [{"parent_candidate_id": "x", "reason": "central text"}]),
        contract(),
    )
    assert report["cohort_salvage_rate"] == 0.5


def test_incomplete_manual_coverage_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate(manifest(), [], post(), materialization(), contract())
