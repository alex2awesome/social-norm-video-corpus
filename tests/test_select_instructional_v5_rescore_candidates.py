from scripts.select_instructional_v5_rescore_candidates import (
    first_stage_pass,
    latest_successes,
    select,
)


def output(item_id, result=None, error=None):
    return {"item_id": item_id, "result": result, "error": error}


def qwen_result(**updates):
    result = {
        "social_norm_domain": "yes",
        "usable_demo_after_relabel": "yes",
        "localization_quality": "clean",
    }
    result.update(updates)
    return result


def test_first_stage_requires_all_four_atoms():
    assert first_stage_pass(
        qwen_result(), {"social_norm_candidate": "yes"}
    )
    assert not first_stage_pass(
        qwen_result(localization_quality="broad"),
        {"social_norm_candidate": "yes"},
    )
    assert not first_stage_pass(
        qwen_result(), {"social_norm_candidate": "no"}
    )


def test_latest_successes_uses_latest_valid_append_only_row():
    rows = [
        output("a", qwen_result()),
        output("a", None, "timeout"),
        output("a", qwen_result(social_norm_domain="no")),
    ]
    assert (
        latest_successes(rows)["a"]["result"]["social_norm_domain"] == "no"
    )


def test_select_preserves_source_fields_and_counts_unique_uids():
    manifest = [
        {"item_id": "a", "uid": "u", "category": "c", "polarity": "correct"},
        {"item_id": "b", "uid": "u", "category": "c", "polarity": "violation"},
        {"item_id": "c", "uid": "v", "category": "d", "polarity": "correct"},
    ]
    qwen = [
        output("a", qwen_result()),
        output("b", qwen_result()),
        output("c", qwen_result(usable_demo_after_relabel="no")),
    ]
    text = [
        output("a", {"social_norm_candidate": "yes"}),
        output("b", {"social_norm_candidate": "yes"}),
        output("c", {"social_norm_candidate": "yes"}),
    ]
    selected, summary = select(manifest, qwen, text)
    assert [row["item_id"] for row in selected] == ["a", "b"]
    assert summary["selected_items"] == 2
    assert summary["selected_unique_uids"] == 1
    assert summary["by_polarity"] == {"correct": 1, "violation": 1}
