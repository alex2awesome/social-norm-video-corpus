import pytest

from scripts.build_instructional_v23_replication_ledger import build


def test_build_marks_only_exact_labels_usable() -> None:
    manifest = [
        {"audit_index": i, "candidate_id": f"c{i}", "item_id": f"i{i}", "uid": f"u{i}"}
        for i in range(3)
    ]
    visual = [
        {"audit_index": "0", "visual_demo": "Y", "visual_form": "scene", "note": "v0"},
        {"audit_index": "1", "visual_demo": "Y", "visual_form": "scene", "note": "v1"},
        {"audit_index": "2", "visual_demo": "N", "visual_form": "lecture", "note": "v2"},
    ]
    semantic = [
        {"audit_index": "0", "semantic_status": "exact", "note": "s0"},
        {"audit_index": "1", "semantic_status": "relabel", "note": "s1"},
    ]
    rows = build(manifest, visual, semantic)
    assert [row["visual_demo"] for row in rows] == ["yes", "yes", "no"]
    assert [row["complete_demo"] for row in rows] == ["yes", "yes", "no"]
    assert [row["usable_demo"] for row in rows] == ["yes", "no", "no"]


def test_build_requires_semantic_review_of_every_visual_positive() -> None:
    with pytest.raises(ValueError, match="exactly the visual positives"):
        build(
            [{"audit_index": 0, "candidate_id": "c", "item_id": "i", "uid": "u"}],
            [{"audit_index": "0", "visual_demo": "Y", "visual_form": "scene", "note": "v"}],
            [],
        )
