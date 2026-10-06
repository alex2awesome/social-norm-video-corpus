import json

import pytest

from scripts.run_open_vlm_scene_benchmark import parse_json


def valid_v9a() -> dict:
    return {
        "observable_event": "yes",
        "event_start_percent": 20,
        "event_end_percent": 70,
        "literal_actor": "a seated customer",
        "literal_action_or_situated_utterance": "asks a waiter for still water",
        "literal_affected_party_or_shared_setting": "the waiter at the table",
        "same_event_actor_action_target": "yes",
        "event_temporally_localized": "yes",
        "evidence_source": "situated_dialogue_or_subtitles",
        "scene_role": "demonstrated_event",
        "demonstration_kind": "explicit_social_etiquette",
        "presentation_only": "no",
        "montage_only": "no",
        "aftermath_only": "no",
        "metadata_needed_to_name_action": "no",
        "evidence_before": "the customer and waiter face each other",
        "evidence_during": "the customer asks for still water",
        "evidence_after": "the waiter pours water",
        "evidence": "A customer asks a waiter for water at a table.",
    }


def test_v9a_accepts_consistent_label_blind_event_record():
    result = valid_v9a()
    assert parse_json(json.dumps(result), "v9a") == result


def test_v9a_fails_closed_when_same_event_requirement_is_missing():
    result = valid_v9a()
    result["same_event_actor_action_target"] = "no"
    parsed = parse_json(json.dumps(result), "v9a")
    assert parsed["observable_event"] == "uncertain"
    assert "fail_closed" in parsed["observable_event_validation_repair"]


def test_v9a_accepts_explicit_etiquette_as_a_distinct_demo_kind():
    result = valid_v9a()
    parsed = parse_json(json.dumps(result), "v9a")
    assert parsed["demonstration_kind"] == "explicit_social_etiquette"
    assert parsed["observable_event"] == "yes"


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (-1, 30, "both be -1"),
        (80, 20, "start must not follow"),
    ],
)
def test_v9a_rejects_invalid_temporal_spans(start, end, message):
    result = valid_v9a()
    result["event_start_percent"] = start
    result["event_end_percent"] = end
    with pytest.raises(ValueError, match=message):
        parse_json(json.dumps(result), "v9a")


def test_v9a_repairs_out_of_range_bounds_fail_closed():
    result = valid_v9a()
    result["event_end_percent"] = 110
    parsed = parse_json(json.dumps(result), "v9a")
    assert parsed["event_start_percent"] == -1
    assert parsed["event_end_percent"] == -1
    assert parsed["observable_event"] == "uncertain"


def test_v9a_repairs_non_gating_glm_surface_variants():
    result = valid_v9a()
    del result["evidence"]
    parsed = parse_json(json.dumps(result), "v9a")
    assert parsed["evidence"] == result["evidence_during"]

    negative = valid_v9a()
    negative["observable_event"] = "no"
    negative["event_start_percent"] = -1
    negative["event_end_percent"] = -1
    negative["same_event_actor_action_target"] = "no"
    negative["event_temporally_localized"] = "no"
    negative["evidence_source"] = "generic_context_broll"
    negative["scene_role"] = "generic_context_broll"
    parsed = parse_json(json.dumps(negative), "v9a")
    assert parsed["evidence_source"] == "generic_motion"
    assert parsed["observable_event"] == "no"
