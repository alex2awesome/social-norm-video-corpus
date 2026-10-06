import pytest

from scripts.evaluate_commentary_clip_plan import evaluate


def manifest():
    return [{
        "candidate_id": "a", "audit_index": 0, "proxy_clip": "/a.mp4",
        "proxy_clip_sha256": "hash",
    }]


def blind(**overrides):
    row = {
        "candidate_id": "a", "performed_event_visible": "yes",
        "actor_target_grounded": "yes", "start_boundary_clean": "yes",
        "end_boundary_clean": "yes", "label_bearing_text_absent": "yes",
        "audio_absent": "yes", "medium": "live_action",
        "blind_evidence": "A person performs an action toward another person.",
    }
    row.update(overrides)
    return [row]


def post(**overrides):
    row = {
        "candidate_id": "a", "exact_named_action_visible": "yes",
        "label_alignment": "exact", "route": "commentary_visual",
        "post_reveal_evidence": "The visible act exactly matches the title.",
    }
    row.update(overrides)
    return [row]


def test_exact_complete_manual_review_creates_individual_acceptance_only():
    report, accepted = evaluate(manifest(), blind(), post())
    assert report["exact_commentary_visual_clips"] == 1
    assert report["automatic_acceptance_rule_promoted"] is False
    assert accepted[0]["manual_acceptance"] == "exact_commentary_visual_clip"
    assert accepted[0]["corpus_disposition"] is None


def test_any_failed_transform_or_semantic_check_fails_closed():
    report, accepted = evaluate(
        manifest(), blind(label_bearing_text_absent="no"), post()
    )
    assert accepted == []
    assert report["failed_check_counts"]["label_bearing_text_absent"] == 1


def test_instructional_demo_is_rerouted_not_commentary_accepted():
    report, accepted = evaluate(manifest(), blind(), post(route="instructional_demo"))
    assert accepted == []
    assert report["route_counts"] == {"instructional_demo": 1}


def test_incomplete_ledger_is_rejected():
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate(manifest(), blind(), [])
