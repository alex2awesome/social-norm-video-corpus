import json

import pytest

from scripts.run_open_vlm_scene_benchmark import parse_json


def valid_result() -> dict:
    return {
        "visually_observable_event": "yes",
        "event_start_percent": 20,
        "event_end_percent": 60,
        "actor_visible_description": "a child",
        "action_or_situated_speech_description": "hands a ball to another child",
        "affected_party_or_shared_setting_description": "another child",
        "evidence_before": "one child holds the ball",
        "evidence_during": "both children hold the ball",
        "evidence_after": "the other child holds the ball",
        "metadata_needed_to_identify_action": "no",
        "presentation_or_context_only": "no",
        "depiction_type": "enacted_scene",
        "confidence": 0.9,
        "evidence": "The ball visibly transfers between two children.",
    }


def test_v6_schema_requires_label_free_temporal_evidence():
    result = valid_result()

    assert parse_json(json.dumps(result), "v6") == result

    del result["evidence_during"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_json(json.dumps(result), "v6")


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (70, 20, "start must not follow"),
        (-1, 20, "both be -1"),
    ],
)
def test_v6_rejects_invalid_temporal_spans(start, end, message):
    result = valid_result()
    result["event_start_percent"] = start
    result["event_end_percent"] = end

    with pytest.raises(ValueError, match=message):
        parse_json(json.dumps(result), "v6")


def test_v6_preserves_event_but_fails_closed_on_out_of_range_bounds():
    result = valid_result()
    result["event_start_percent"] = 170
    result["event_end_percent"] = 175

    parsed = parse_json(json.dumps(result), "v6")

    assert parsed["visually_observable_event"] == "yes"
    assert parsed["event_start_percent"] == -1
    assert parsed["event_end_percent"] == -1
    assert (
        parsed["temporal_bounds_validation_repair"]
        == "out_of_range_to_unlocalized"
    )


def test_v6_visible_event_cannot_omit_span():
    result = valid_result()
    result["event_start_percent"] = -1
    result["event_end_percent"] = -1

    with pytest.raises(ValueError, match="require a temporal span"):
        parse_json(json.dumps(result), "v6")
