from scripts.evaluate_witnessed_creator_initiated_audit import evaluate


def review_rows():
    sealed = [
        {
            "audit_index": 0,
            "item_id": "a",
            "cue_group": "creator_initiated_title",
            "title": "Trying to kiss strangers",
        },
        {
            "audit_index": 1,
            "item_id": "b",
            "cue_group": "creator_initiated_title",
            "title": "Man touching women without consent",
        },
    ]
    blind = [
        {
            "audit_index": index,
            "item_id": item,
            "visual_social_interaction": "yes",
            "visually_produced_or_scripted": "uncertain",
            "potential_instructional_demo": "yes",
            "visual_form": "scene",
            "evidence_note": "Visible interaction.",
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
            "evidence_note": "Creator initiated.",
            "review_complete": "yes",
        },
        {
            "audit_index": 1,
            "item_id": "b",
            "explicit_staging_confirmed":"no",
            "deliberately_produced_violation_setup":"uncertain",
            "organic_witnessed_eligible":"no",
            "instructional_candidate":"yes",
            "recommended_route":"manual_visual_candidate",
            "evidence_note":"Descriptive report title.",
            "review_complete":"yes"
        },
    ]
    return sealed, blind, post


def test_refined_rule_removes_descriptive_touching_false_positive():
    report = evaluate(*review_rows())
    rule = report["refined_rule"]
    assert rule["tp"] == 1
    assert rule["fp"] == 0
    assert rule["tn"] == 1
    assert rule["excluded_false_positive_item_ids"] == ["b"]


def test_small_audit_cannot_promote_rule():
    report = evaluate(*review_rows())
    assert report["decision"]["promote_refined_cue"] is False
