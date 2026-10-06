import pytest

from scripts.evaluate_witnessed_staging_audit import evaluate


def rows():
    sealed = [
        {"audit_index": 0, "item_id": "a", "cue_group": "title"},
        {"audit_index": 1, "item_id": "b", "cue_group": "creator"},
    ]
    blind = [
        {
            "audit_index": index,
            "item_id": item,
            "visual_social_interaction": "yes",
            "visually_produced_or_scripted": "yes",
            "potential_instructional_demo": "yes",
            "visual_form": "scene",
            "evidence_note": "Visible scene.",
            "review_complete": "yes",
        }
        for index, item in enumerate(("a", "b"))
    ]
    post = [
        {
            "audit_index": 0,
            "item_id": "a",
            "explicit_staging_confirmed": "yes",
            "deliberately_produced_violation_setup": "yes",
            "organic_witnessed_eligible": "no",
            "instructional_candidate": "yes",
            "recommended_route": "staged_instructional_candidate",
            "evidence_note": "Explicit prank.",
            "review_complete": "yes",
        },
        {
            "audit_index": 1,
            "item_id": "b",
            "explicit_staging_confirmed": "no",
            "deliberately_produced_violation_setup": "no",
            "organic_witnessed_eligible": "yes",
            "instructional_candidate": "yes",
            "recommended_route": "witnessed_candidate_reaudit",
            "evidence_note": "Generic creator intro.",
            "review_complete": "yes",
        },
    ]
    return sealed, blind, post


def test_evaluate_title_rule():
    report = evaluate(*rows())
    metric = report["rules"]["title_prank_social_experiment_hidden_camera"]
    assert metric["precision"] == 1.0
    assert metric["recall"] == 1.0
    assert report["decision"]["promote_title_cue"] is True


def test_validate_rejects_identity_mismatch():
    sealed, blind, post = rows()
    blind[0]["item_id"] = "wrong"
    with pytest.raises(ValueError, match="coverage"):
        evaluate(sealed, blind, post)
