import copy

import pytest

from scripts.apply_commentary_manual_text_review_v2 import validate_and_merge


def source():
    return {
        "uid": "youtube__abc",
        "item_id": "commentary:youtube__abc:0",
        "transcript_context": [{"start": 1, "end": 3, "text": "A customer hit the worker. That was cruel."}],
    }


def review():
    return {
        "uid": "youtube__abc",
        "item_id": "commentary:youtube__abc:0",
        "decision": "accept_after_relabel",
        "social_actor_grounded": "yes",
        "concrete_behavior": "yes",
        "target_or_shared_context_grounded": "yes",
        "normative_stance_grounded": "yes",
        "normalized_behavior": "a customer hits a worker",
        "normalized_norm": "do not assault workers",
        "behavior_evidence_quote": "A customer hit the worker",
        "stance_evidence_quote": "That was cruel",
        "manual_rationale": "Actor, action, target, and judgment are explicit.",
    }


def test_accepts_complete_exact_review_without_claiming_visual_evidence():
    merged, summary = validate_and_merge([source()], [review()])
    assert summary["coverage"] == 1
    assert summary["accepted_for_visual_search"] == 1
    assert merged[0]["text_label_manual_reviewed"] is True
    assert merged[0]["script_certifies_visual_event"] is False


def test_rejects_partial_coverage():
    with pytest.raises(ValueError, match="exactly cover"):
        validate_and_merge([source()], [])


def test_rejects_accepted_label_without_all_atomic_yes_fields():
    bad = copy.deepcopy(review())
    bad["concrete_behavior"] = "no"
    with pytest.raises(ValueError, match="accepted row"):
        validate_and_merge([source()], [bad])


def test_rejects_ungrounded_evidence_quote():
    bad = copy.deepcopy(review())
    bad["behavior_evidence_quote"] = "something never said"
    with pytest.raises(ValueError, match="not grounded"):
        validate_and_merge([source()], [bad])
