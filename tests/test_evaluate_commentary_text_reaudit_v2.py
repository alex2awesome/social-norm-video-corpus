import copy

import pytest

from scripts.evaluate_commentary_text_reaudit_v2 import validate


SHA = "a" * 64


def source(item_id="x", decision="accept_after_relabel"):
    return {"item_id": item_id, "decision": decision}


def review(item_id="x"):
    return {
        "item_id": item_id,
        "v1_decision": "accept_after_relabel",
        "strict_event_label_decision": "accept",
        "occurrence_grounding": "bounded_occurrence",
        "behavior_specificity": "specific",
        "stance_quality": "unambiguous_external",
        "visual_search_route": "strict_visual_search",
        "corrected_behavior": "one person takes another person's wallet",
        "corrected_norm": "do not steal",
        "agrees_with_v1": True,
        "manual_rationale": "The transcript grounds an occurred action and criticism.",
    }


def test_accepts_exact_complete_strict_review():
    result = validate([source()], [review()], SHA, SHA)
    assert result["coverage"] == 1
    assert result["strict_event_accepted"] == 1
    assert result["v1_accept_retention"] == 1
    assert result["automatic_acceptance"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("occurrence_grounding", "generic_or_hypothetical"),
        ("behavior_specificity", "underspecified"),
        ("stance_quality", "contested"),
        ("visual_search_route", "exploratory_ambiguous"),
    ],
)
def test_strict_accept_requires_every_strict_invariant(field, value):
    row = review()
    row[field] = value
    with pytest.raises(ValueError):
        validate([source()], [row], SHA, SHA)


def test_reject_can_preserve_generic_instructional_retrieval_lead():
    row = review()
    row.update(
        strict_event_label_decision="reject",
        occurrence_grounding="generic_or_hypothetical",
        visual_search_route="instructional_definition_retrieval",
        corrected_behavior="",
        corrected_norm="",
        agrees_with_v1=False,
    )
    result = validate([source()], [row], SHA, SHA)
    assert result["strict_event_accepted"] == 0
    assert result["routes"] == {"instructional_definition_retrieval": 1}


def test_rejects_hash_lineage_mismatch():
    with pytest.raises(ValueError, match="hash"):
        validate([source()], [review()], SHA, "b" * 64)


def test_rejects_coverage_or_v1_lineage_mismatch():
    with pytest.raises(ValueError, match="exactly cover"):
        validate([source()], [review("y")], SHA, SHA)
    row = copy.deepcopy(review())
    row["v1_decision"] = "reject"
    with pytest.raises(ValueError, match="lineage"):
        validate([source()], [row], SHA, SHA)


def test_rejects_false_agreement_claim():
    row = review()
    row["agrees_with_v1"] = False
    with pytest.raises(ValueError, match="agreement"):
        validate([source()], [row], SHA, SHA)
