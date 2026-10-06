import copy

import pytest

from scripts.freeze_instructional_social_text_gold_v5 import freeze


SHA = "a" * 64


def packet():
    return {
        "blind_id": "instructional-social-v5-0000",
        "audit_index": 0,
        "proposed_norm": "set a calm boundary",
        "proposed_polarity": "correct",
        "start_quote": "If this continues, I may need to end the call.",
        "end_quote": "end the call",
        "explanation": "This names the consequence without being defensive.",
    }


def gold():
    return {
        "blind_id": "instructional-social-v5-0000",
        "audit_index": 0,
        "manual_reviewed": True,
        "prior_visual_or_model_labels_revealed": False,
        "social_behavior_domain": "communicative_act",
        "specific_actor_role_grounded": "yes",
        "specific_behavior_grounded": "yes",
        "affected_person_or_shared_social_context_grounded": "yes",
        "normative_obligation_grounded": "yes",
        "literal_social_behavior_not_analogy": "yes",
        "procedural_skill_only": "no",
        "personal_optimization_only": "no",
        "proposed_norm_supported": "yes",
        "proposed_polarity_supported": "yes",
        "normalized_behavior": "a caller states a consequence",
        "normalized_norm": "set a calm boundary",
        "behavior_evidence_quote": "I may need to end the call",
        "norm_evidence_quote": "without being defensive",
        "route": "strict_social_alignment",
        "description": "A literal boundary-setting utterance and approving explanation are grounded.",
    }


def test_freezes_exact_complete_grounded_blind_gold():
    result = freeze([packet()], [gold()], SHA, SHA, "b" * 64, expected_items=1)
    assert result["coverage"] == 1
    assert result["strict_social_alignment"] == 1
    assert result["literal_social_candidates_after_relabel"] == 1
    assert result["visual_demo_certified"] is False
    assert result["automatic_acceptance"] is False


def test_rejects_hash_coverage_or_reveal_leak():
    with pytest.raises(ValueError, match="hash"):
        freeze([packet()], [gold()], "c" * 64, SHA, "b" * 64, expected_items=1)
    with pytest.raises(ValueError, match="count"):
        freeze([packet()], [gold()], SHA, SHA, "b" * 64, expected_items=2)
    row = gold()
    row["title"] = "revealed title"
    with pytest.raises(ValueError, match="leaked"):
        freeze([packet()], [row], SHA, SHA, "b" * 64, expected_items=1)


def test_rejects_ungrounded_quote_or_revealed_prior_label():
    row = copy.deepcopy(gold())
    row["behavior_evidence_quote"] = "a cyclist smashed a mirror"
    with pytest.raises(ValueError, match="not grounded"):
        freeze([packet()], [row], SHA, SHA, "b" * 64, expected_items=1)
    row = gold()
    row["prior_visual_or_model_labels_revealed"] = True
    with pytest.raises(ValueError, match="revealed"):
        freeze([packet()], [row], SHA, SHA, "b" * 64, expected_items=1)
