from scripts.select_instructional_v18_animation_holdout import select


def model(item: str, **result) -> dict:
    return {"item_id": item, "error": None, "result": result}


def test_v18_selects_animation_and_two_disjoint_control_bands() -> None:
    source = [
        {"item_id": f"i{i}", "uid": f"u{i}", "polarity": "violation"}
        for i in range(4)
    ]
    boards = [
        {
            "item_id": f"i{i}",
            "uid": f"u{i}",
            "sheet_path": f"{i}.jpg",
            "sheet_sha256": f"h{i}",
            "sampled_timestamps": [0, 10],
        }
        for i in range(4)
    ]
    glm = {
        "i0": model(
            "i0", demo_usable="yes", scene_role="animation_or_story_demo"
        ),
        "i1": model("i1", demo_usable="yes", scene_role="situated_scene"),
        "i2": model("i2", demo_usable="yes", scene_role="situated_scene"),
        "i3": model("i3", demo_usable="no", scene_role="situated_scene"),
    }
    qwen = {
        "i0": model("i0", scene_role="animation_or_story_demo"),
        "i1": model("i1", scene_role="situated_scene"),
        "i2": model("i2", scene_role="presenter_sample_or_advice"),
        "i3": model("i3", scene_role="situated_scene"),
    }
    gemma = {
        f"i{i}": model(f"i{i}", visual_record_supports_concrete_demo="yes")
        for i in range(4)
    }
    semantic, blind, summary = select(
        source, boards, glm, qwen, gemma, set(), 30, 15
    )
    assert {row["band"] for row in semantic} == {
        "candidate",
        "live_scene_control",
        "qwen_scene_reject_control",
    }
    assert len(blind) == 3
    assert all(set(row) == {
        "audit_index",
        "candidate_id",
        "sheet_path",
        "sheet_sha256",
    } for row in blind)
    assert summary["population_by_band"] == {
        "candidate": 1,
        "live_scene_control": 1,
        "qwen_scene_reject_control": 1,
    }
