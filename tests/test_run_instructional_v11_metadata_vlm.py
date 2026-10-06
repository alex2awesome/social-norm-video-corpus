import json

from scripts.run_instructional_v11_metadata_vlm import parse_result


def _record() -> dict:
    return {
        "performed_social_event": "yes",
        "actor_action_affected_party_same_event": "yes",
        "affected_party_present_in_event": "yes",
        "performed_violation_not_only_described": "yes",
        "proposed_actor_behavior_target_match": "yes",
        "event_polarity": "violation",
        "proposed_norm_relation": "exact",
        "interaction_evidence": "co_present_action_or_dialogue",
        "demo_format": "animation",
        "exact_violation_demo": "yes",
        "rejection_reason": "none",
        "literal_event": "child takes another child's toy",
        "evidence": "The other child is present and reacts.",
    }


def test_parse_accepts_strict_exact_violation() -> None:
    parsed = parse_result(json.dumps(_record()))
    assert parsed["exact_violation_demo"] == "yes"


def test_parse_fails_closed_when_violation_is_only_prohibited() -> None:
    record = _record()
    record["performed_violation_not_only_described"] = "no"
    record["event_polarity"] = "explanation_or_prohibition"
    record["rejection_reason"] = "only_described_or_prohibited"
    parsed = parse_result(json.dumps(record))
    assert parsed["exact_violation_demo"] == "uncertain"
    assert "inconsistent_positive" in parsed["validation_repairs"]


def test_parse_fails_closed_on_unknown_enum() -> None:
    record = _record()
    record["proposed_norm_relation"] = "sort of"
    parsed = parse_result(json.dumps(record))
    assert parsed["proposed_norm_relation"] == "uncertain"
    assert parsed["exact_violation_demo"] == "uncertain"


def test_parse_repairs_null_literal_on_negative() -> None:
    record = _record()
    record["literal_event"] = None
    record["exact_violation_demo"] = "no"
    parsed = parse_result(json.dumps(record))
    assert parsed["literal_event"] == "none"
    assert parsed["exact_violation_demo"] == "no"


def test_parse_fails_closed_on_null_literal_positive() -> None:
    record = _record()
    record["literal_event"] = None
    parsed = parse_result(json.dumps(record))
    assert parsed["exact_violation_demo"] == "uncertain"
