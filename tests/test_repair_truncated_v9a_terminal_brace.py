import json

import pytest

from scripts.repair_truncated_v9a_terminal_brace import repair


def failed(raw: str) -> dict:
    return {
        "item_id": "x",
        "rubric": "v9a_storyboard",
        "raw_response": raw,
        "result": None,
        "error": "ValueError: response contained no JSON object",
    }


def valid_payload() -> dict:
    return {
        "observable_event": "yes",
        "event_start_percent": 10,
        "event_end_percent": 50,
        "literal_actor": "person",
        "literal_action_or_situated_utterance": "touches another person",
        "literal_affected_party_or_shared_setting": "other person",
        "same_event_actor_action_target": "yes",
        "event_temporally_localized": "yes",
        "evidence_source": "physical_action",
        "scene_role": "demonstrated_event",
        "demonstration_kind": "interpersonal_conduct",
        "presentation_only": "no",
        "montage_only": "no",
        "aftermath_only": "no",
        "metadata_needed_to_name_action": "no",
        "evidence_before": "people stand",
        "evidence_during": "one touches another",
        "evidence_after": "they separate",
        "evidence": "visible contact",
    }


def test_repairs_only_terminal_brace_and_preserves_raw():
    raw = json.dumps(valid_payload())[:-1]
    result = repair(failed(raw))
    assert result["error"] is None
    assert result["result"]["observable_event"] == "yes"
    assert result["raw_response"] == raw
    assert not result["repair_provenance"]["semantic_fields_changed"]


def test_refuses_already_closed_or_structurally_incomplete_json():
    with pytest.raises(ValueError, match="unterminated"):
        repair(failed(json.dumps(valid_payload())))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        repair(failed('{"observable_event": "yes"'))
