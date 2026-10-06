import pytest

from scripts.render_title_event_dense_followup import select_followups


def test_select_followups_preserves_audit_order_and_only_unresolved():
    candidates = [
        {"item_id": "a", "uid": "ua", "norm": "must not leak"},
        {"item_id": "b", "uid": "ub", "norm": "must not leak"},
    ]
    blind = [
        {"audit_index": 1, "item_id": "b", "uid": "ub", "pillar": "commentary"},
        {"audit_index": 0, "item_id": "a", "uid": "ua", "pillar": "commentary"},
    ]
    post = [
        {"audit_index": 0, "disposition": "dense_review"},
        {"audit_index": 1, "disposition": "title_labeled_visual_event"},
    ]
    selected = select_followups(candidates, blind, post)
    assert [row[1]["item_id"] for row in selected] == ["a"]


def test_select_followups_requires_complete_coverage():
    with pytest.raises(ValueError, match="coverage mismatch"):
        select_followups(
            [{"item_id": "a", "uid": "ua"}],
            [{"audit_index": 0, "item_id": "a", "uid": "ua"}],
            [],
        )


def test_select_followups_checks_candidate_identity():
    with pytest.raises(ValueError, match="identity mismatch"):
        select_followups(
            [{"item_id": "a", "uid": "wrong"}],
            [{"audit_index": 0, "item_id": "a", "uid": "ua"}],
            [{"audit_index": 0, "disposition": "crop_review"}],
        )
