from scripts.sample_instructional_v5_validation import (
    balanced_source_disjoint,
    select_controls,
    select_final,
)


def vote(strict=True):
    value = {
        "social_norm_domain": "yes",
        "behavior_occurs_in_scene": "yes",
        "affected_party_or_shared_setting_visible": "yes",
        "situated_interaction_complete": "yes",
        "usable_demo_after_relabel": "yes",
    }
    if not strict:
        value["behavior_occurs_in_scene"] = "no"
    return value


def output(item_id, strict=True):
    return {"item_id": item_id, "result": vote(strict), "error": None}


def row(item_id, uid, category="c", polarity="correct"):
    return {
        "item_id": item_id,
        "uid": uid,
        "category": category,
        "polarity": polarity,
    }


def test_balanced_sample_is_uid_disjoint_and_deterministic():
    rows = [
        row("a", "u1", "a"),
        row("b", "u1", "b"),
        row("c", "u2", "a"),
        row("d", "u3", "b"),
    ]
    first = balanced_source_disjoint(rows, 3, 7)
    second = balanced_source_disjoint(rows, 3, 7)
    assert [value["item_id"] for value in first] == [
        value["item_id"] for value in second
    ]
    assert len({value["uid"] for value in first}) == 3


def test_controls_are_frozen_from_qwen_rejects_only():
    manifest = [row("a", "u1"), row("b", "u2"), row("c", "u3")]
    qwen = [output("a"), output("b", False), output("c", False)]
    controls = select_controls(manifest, qwen, {"u2"}, 5, 9)
    assert [value["item_id"] for value in controls] == ["c"]
    assert controls[0]["audit_band"] == "qwen_reject_control"


def test_final_selects_bands_and_preserves_controls():
    manifest = [
        row("a", "u1"),
        row("b", "u2"),
        row("c", "u3"),
        row("d", "u4"),
    ]
    qwen = [
        output("a"),
        output("b"),
        output("c", False),
        output("d"),
    ]
    glm = [
        output("a"),
        output("b", False),
        output("c"),
        output("d"),
    ]
    controls = [
        {
            **row("c", "u3"),
            "audit_band": "qwen_reject_control",
            "qwen_v5_conditioned_result": vote(False),
        }
    ]
    selected = select_final(
        manifest,
        qwen,
        glm,
        controls,
        set(),
        dual_count=1,
        disagreement_count=1,
        seed=2,
    )
    assert [value["audit_band"] for value in selected] == [
        "dual_strict",
        "qwen_strict_glm_reject",
        "qwen_reject_control",
    ]
    assert len({value["uid"] for value in selected}) == 3
    assert [value["audit_index"] for value in selected] == [0, 1, 2]
