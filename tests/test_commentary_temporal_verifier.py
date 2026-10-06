import json
import io
import urllib.error

from scripts.run_commentary_temporal_verifier_vlm import (
    exception_detail,
    parse_stage_a,
    parse_stage_a_v3,
    parse_stage_b,
    parse_stage_b_v3,
    rubric_name,
    stage_a_system,
    strict_pass,
    strict_pass_v3,
)


def stage_a():
    return {
        "sequence_contains_observable_event": "yes",
        "actor_visible": "yes",
        "target_or_property_visible": "yes",
        "before_action_after_complete": "yes",
        "crucial_action_occluded_or_offframe": "no",
        "caption_or_narration_needed": "no",
        "observable_action_kind": "property_transfer",
        "literal_before": "Person approaches empty-handed.",
        "literal_action": "Person lifts a box.",
        "literal_after": "Person departs carrying the box.",
        "evidence": "The box changes from ground to the person's hands.",
    }


def stage_b():
    return {
        "claim_action_in_literal_sequence": "yes",
        "same_actor_action_target": "yes",
        "claim_requires_unseen_intent_or_ownership": "no",
        "literal_sequence_contradicts_claim": "no",
        "decision": "pass",
        "evidence": "The claimed pickup is directly shown.",
    }


def stage_a_v3():
    return {
        **stage_a(),
        "literal_actor": "Person",
        "literal_target": "box",
        "literal_result": "The box is carried away.",
    }


def stage_b_v3():
    return {
        "core_action_in_literal_sequence": "yes",
        "core_target_and_result_in_literal_sequence": "yes",
        "stage_a_text_explicitly_names_core_action": "yes",
        "semantic_qualifiers_require_external_label": "yes",
        "literal_sequence_contradicts_core_action": "no",
        "decision": "pass",
        "evidence": "Stage A explicitly says the person lifts and carries the box.",
    }


def test_parsers_and_strict_pass_accept_complete_transition():
    a = parse_stage_a(json.dumps(stage_a()))
    b = parse_stage_b(json.dumps(stage_b()))
    assert strict_pass(a, b)


def test_strict_pass_rejects_occluded_action():
    a = stage_a()
    a["crucial_action_occluded_or_offframe"] = "yes"
    assert not strict_pass(a, stage_b())


def test_strict_pass_rejects_unseen_ownership():
    b = stage_b()
    b["claim_requires_unseen_intent_or_ownership"] = "yes"
    assert not strict_pass(stage_a(), b)


def test_stage_a_prompt_uses_manifest_layout_and_describes_redaction():
    prompt = stage_a_system(
        {
            "grid_columns": 4,
            "grid_rows": 4,
            "fps": 4.0,
            "ocr_masked": True,
        }
    )
    assert "4-column by\n4-row" in prompt
    assert "Flat neutral patches cover OCR-detected text" in prompt
    assert rubric_name({"ocr_masked": True}).endswith("_v2_ocr_masked")


def test_stage_a_prompt_keeps_v1_defaults():
    prompt = stage_a_system({})
    assert "8-column by\n4-row" in prompt
    assert "Flat neutral patches" not in prompt
    assert rubric_name({}) == "commentary_temporal_verifier_v1"


def test_core_event_v3_accepts_physical_taking_with_external_ownership_label():
    a = parse_stage_a_v3(json.dumps(stage_a_v3()))
    b = parse_stage_b_v3(json.dumps(stage_b_v3()))
    assert strict_pass_v3(a, b)


def test_core_event_v3_rejects_when_stage_a_does_not_explicitly_name_action():
    b = stage_b_v3()
    b["stage_a_text_explicitly_names_core_action"] = "no"
    assert not strict_pass_v3(stage_a_v3(), b)


def test_core_event_v3_prompt_requires_structured_target_and_result():
    prompt = stage_a_system(
        {
            "grid_columns": 4,
            "grid_rows": 4,
            "fps": 4.0,
            "ocr_masked": True,
        },
        policy="core_event_v3",
    )
    assert "literal_target:" in prompt
    assert "literal_result:" in prompt
    assert rubric_name(
        {"ocr_masked": True},
        policy="core_event_v3",
    ) == "commentary_temporal_core_event_v3_ocr_masked"


def test_http_error_detail_retains_bounded_server_body():
    error = urllib.error.HTTPError(
        "http://localhost/v1/chat/completions",
        400,
        "Bad Request",
        {},
        io.BytesIO(b'{"error":{"message":"prompt too long"}}'),
    )
    detail = exception_detail(error)
    assert "HTTP Error 400" in detail
    assert "prompt too long" in detail
