from scripts.run_open_vlm_v16_social_scene import SYSTEM_V16, v16_base_record


def test_v16_prompt_targets_observed_transfer_failures() -> None:
    prompt = SYSTEM_V16.lower()
    assert "do not equate" in prompt
    assert "talk show" in prompt
    assert "product-labeling" in prompt
    assert "ordinary conversation" in prompt
    assert "reckless driving causes a crash" in prompt
    assert "fail closed" in prompt


def test_v16_record_has_distinct_rubric() -> None:
    row = {
        "item_id": "opaque",
        "pillar": "instructional",
        "sheet_sha256": "a" * 64,
        "frame_count": 36,
    }
    assert v16_base_record(row, "model")["rubric"] == "v16_social_scene"
