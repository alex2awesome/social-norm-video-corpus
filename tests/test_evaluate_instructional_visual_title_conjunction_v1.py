import pytest

from scripts.evaluate_instructional_visual_title_conjunction_v1 import evaluate


def semantic(index, title):
    return {"item_id": f"i{index}", "uid": f"u{index}", "title": title}


def demo(index, visual):
    return {"item_id": f"i{index}", "manual_visual_demo": visual}


def v23(index, consensus, usable):
    return {
        "item_id": f"i{index}", "visual_consensus": consensus,
        "manual_usable_demo": usable,
    }


def test_conjunction_requires_both_independent_fields() -> None:
    rows, report = evaluate(
        [semantic(0, "A roleplay scenario"), semantic(1, "A roleplay scenario"), semantic(2, "A lecture")],
        [demo(0, True), demo(1, False), demo(2, True)],
        [v23(0, True, True), v23(1, False, False), v23(2, True, True)],
    )
    assert [row["visual_title_conjunction"] for row in rows] == [True, False, False]
    assert report["metrics"]["visual_demo_any_valid_medium"]["selected"] == 1
    assert report["metrics"]["visual_demo_any_valid_medium"]["precision"] == 1.0
    assert report["allowed_uses"] == []


def test_join_requires_exact_item_coverage() -> None:
    with pytest.raises(ValueError, match="same items"):
        evaluate([semantic(0, "roleplay")], [demo(0, True)], [])


def test_duplicate_sources_are_rejected() -> None:
    semantics = [semantic(0, "roleplay"), semantic(1, "roleplay")]
    semantics[1]["uid"] = semantics[0]["uid"]
    with pytest.raises(ValueError, match="source-disjoint"):
        evaluate(
            semantics, [demo(0, True), demo(1, True)],
            [v23(0, True, True), v23(1, True, True)],
        )
