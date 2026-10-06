from scripts.evaluate_dual_conditioned_instructional_rule import (
    confusion,
    evaluate,
    latest_successes,
    strict_visual_pass,
)


def visual_result(value: str = "yes") -> dict:
    return {
        "social_norm_domain": value,
        "behavior_occurs_in_scene": value,
        "affected_party_or_shared_setting_visible": value,
        "situated_interaction_complete": value,
        "usable_demo_after_relabel": value,
    }


def output(item_id: str, result: dict | None, error=None) -> dict:
    return {"item_id": item_id, "result": result, "error": error}


def test_latest_successes_ignores_failure_and_keeps_latest_valid():
    rows = [
        output("a", visual_result()),
        output("a", None, "timeout"),
        output("a", visual_result("no")),
    ]
    assert latest_successes(rows)["a"]["result"]["social_norm_domain"] == "no"


def test_strict_visual_pass_fails_closed():
    assert strict_visual_pass(visual_result())
    missing = visual_result()
    missing.pop("situated_interaction_complete")
    assert not strict_visual_pass(missing)


def test_confusion_handles_all_cells():
    result = confusion(
        [True, True, False, False],
        [True, False, True, False],
    )
    assert result["true_positive"] == 1
    assert result["true_negative"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["precision"] == 0.5


def test_evaluate_uses_triple_intersection_and_append_only_amendment():
    manifest = [
        {
            "audit_index": 1,
            "item_id": "a",
            "uid": "u1",
            "gold_usable": False,
        },
        {
            "audit_index": 2,
            "item_id": "b",
            "uid": "u2",
            "gold_usable": True,
        },
    ]
    qwen = [output("a", visual_result()), output("b", visual_result())]
    glm = [output("a", visual_result()), output("b", visual_result("no"))]
    text = [
        output("a", {"social_norm_candidate": "yes"}),
        output("b", {"social_norm_candidate": "yes"}),
    ]
    report, selected = evaluate(
        manifest,
        qwen,
        glm,
        text,
        [{"audit_index": 1, "amended_gold_usable": True}],
    )
    assert [row["item_id"] for row in selected] == ["a"]
    assert report["frozen_original_gold"]["false_positive"] == 1
    assert report["append_only_amended_gold"]["true_positive"] == 1
    assert report["coverage"]["joint_valid"] == 2
