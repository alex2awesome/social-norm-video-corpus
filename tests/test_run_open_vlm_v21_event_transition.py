from scripts import run_open_vlm_v21_event_transition as v21


def test_v21_prompt_is_label_blind_and_demands_temporal_event() -> None:
    prompt = v21.SYSTEM_V21
    assert "LABEL-BLIND" in prompt
    assert "earlier ... then ... later" in prompt
    assert "Animation is not automatically a demo" in prompt
    assert "dashcam" in prompt
    assert "gameplay" in prompt
    assert "presenter" in prompt


def test_v21_record_preserves_identity_and_sets_rubric() -> None:
    row = {
        "item_id": "instructional:u:0",
        "pillar": "instructional",
        "sheet_sha256": "abc",
        "frame_count": 36,
        "candidate_id": "opaque-1",
        "audit_index": 2,
        "title": "must not leak",
    }
    record = v21.v21_base_record(row, "model")
    assert record["rubric"] == "v21_event_transition"
    assert record["candidate_id"] == "opaque-1"
    assert "title" not in record
