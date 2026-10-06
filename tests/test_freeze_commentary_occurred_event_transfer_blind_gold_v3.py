import copy

import pytest

from scripts.freeze_commentary_occurred_event_transfer_blind_gold_v3 import freeze


SHA = "a" * 64


def packet():
    return {
        "blind_id": "b0",
        "transfer_index": 0,
        "transcript_context": [{"text": "A driver entered the bike lane. That was unsafe."}],
    }


def gold():
    return {
        "blind_id": "b0",
        "transfer_index": 0,
        "manual_reviewed": True,
        "prior_label_revealed": False,
        "is_social_norm": "yes",
        "occurred_event_grounded": "yes",
        "social_actor_grounded": "yes",
        "behavior_semantically_specific": "yes",
        "target_or_shared_context_grounded": "yes",
        "normative_stance_grounded": "yes",
        "event_scope": "bounded_occurrence",
        "stance_quality": "unambiguous_external",
        "behavior_evidence_quote": "driver entered the bike lane",
        "stance_evidence_quote": "That was unsafe",
        "normalized_behavior": "a driver enters a bike lane",
        "normalized_norm": "keep vehicles out of bike lanes",
        "route": "strict_visual_search",
        "description": "Occurred maneuver and criticism are transcript-grounded.",
    }


def test_freezes_complete_grounded_blind_gold():
    result = freeze([packet()], [gold()], SHA, SHA)
    assert result["coverage"] == 1
    assert result["strict_event_candidates"] == 1
    assert result["prior_labels_revealed"] is False


def test_rejects_unsealed_packet_or_prior_label_leak():
    with pytest.raises(ValueError, match="hash"):
        freeze([packet()], [gold()], "b" * 64, SHA)
    row = gold()
    row["v1_decision"] = "accept"
    with pytest.raises(ValueError, match="leaked"):
        freeze([packet()], [row], SHA, SHA)


def test_rejects_ungrounded_evidence_quote():
    row = copy.deepcopy(gold())
    row["behavior_evidence_quote"] = "a cyclist smashed a mirror"
    with pytest.raises(ValueError, match="not grounded"):
        freeze([packet()], [row], SHA, SHA)
