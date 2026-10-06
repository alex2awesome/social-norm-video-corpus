from scripts.select_instructional_v5_strict_candidates import (
    latest_successes,
    partition,
    strict_v5_pass,
)


def result(**updates):
    value = {
        "social_norm_domain": "yes",
        "behavior_occurs_in_scene": "yes",
        "affected_party_or_shared_setting_visible": "yes",
        "situated_interaction_complete": "yes",
        "usable_demo_after_relabel": "yes",
    }
    value.update(updates)
    return value


def output(item_id, value=None, error=None):
    return {"item_id": item_id, "result": value, "error": error}


def test_strict_v5_requires_every_positive_atom():
    assert strict_v5_pass(result())
    for field in result():
        assert not strict_v5_pass(result(**{field: "uncertain"}))


def test_latest_successes_ignores_failed_attempt_but_uses_later_success():
    rows = [
        output("a", result()),
        output("a", None, "timeout"),
        output("a", result(behavior_occurs_in_scene="no")),
    ]
    assert latest_successes(rows)["a"]["result"]["behavior_occurs_in_scene"] == "no"


def test_partition_reports_strict_intersection_and_disagreements():
    manifest = [
        {"item_id": "a", "uid": "u1", "category": "c", "polarity": "correct"},
        {"item_id": "b", "uid": "u2", "category": "c", "polarity": "violation"},
        {"item_id": "c", "uid": "u3", "category": "d", "polarity": "correct"},
        {"item_id": "d", "uid": "u4", "category": "d", "polarity": "violation"},
    ]
    qwen = [
        output("a", result()),
        output("b", result()),
        output("c", result(behavior_occurs_in_scene="no")),
        output("d", result(behavior_occurs_in_scene="no")),
    ]
    glm = [
        output("a", result()),
        output("b", result(situated_interaction_complete="no")),
        output("c", result()),
        output("d", result(behavior_occurs_in_scene="no")),
    ]
    qwen_strict, dual_strict, summary = partition(manifest, qwen, glm)
    assert [row["item_id"] for row in qwen_strict] == ["a", "b"]
    assert [row["item_id"] for row in dual_strict] == ["a"]
    assert summary["bands"] == {
        "dual_reject": 1,
        "dual_strict": 1,
        "glm_only": 1,
        "qwen_only": 1,
    }
    assert summary["dual_by_category"] == {"c": 1}


def test_partition_without_glm_emits_qwen_queue_only():
    manifest = [
        {"item_id": "a", "uid": "u1"},
        {"item_id": "b", "uid": "u2"},
    ]
    qwen = [
        output("a", result()),
        output("b", result(social_norm_domain="no")),
    ]
    qwen_strict, dual_strict, summary = partition(manifest, qwen)
    assert [row["item_id"] for row in qwen_strict] == ["a"]
    assert dual_strict == []
    assert summary["bands"] == {"qwen_reject": 1, "qwen_strict": 1}
