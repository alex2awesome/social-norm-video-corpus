import json

from scripts.run_open_vlm_scene_benchmark import parse_json


def valid_v8():
    return {
        "on_screen_social_event": "yes",
        "actor_performs_target_behavior": "yes",
        "affected_party_or_shared_context_same_event": "yes",
        "behavior_socially_evaluable_from_clip": "yes",
        "event_temporally_localized": "yes",
        "informal_social_conduct_not_formal_procedure": "yes",
        "evidence_source": "situated_dialogue_or_subtitles",
        "visual_role": "demonstrated_event",
        "usable_demo_after_relabel": "yes",
        "proposed_norm_supported": "yes",
        "literal_actor": "a customer",
        "literal_action": "asks a worker for help",
        "literal_affected_party_or_shared_context": "the worker at a store counter",
        "rejection_reason": "none",
        "evidence": "A customer asks a worker for help at the counter.",
    }


def test_v8_accepts_internally_consistent_strict_result():
    result = valid_v8()
    assert parse_json(json.dumps(result), "v8") == result


def test_v8_fails_closed_on_usable_yes_when_context_only():
    result = valid_v8()
    result["visual_role"] = "generic_context_broll"
    parsed = parse_json(json.dumps(result), "v8")
    assert parsed["usable_demo_after_relabel"] == "uncertain"
    assert "fail_closed" in parsed["usable_demo_validation_repair"]


def test_v8_allows_fail_closed_negative():
    result = valid_v8()
    result["actor_performs_target_behavior"] = "no"
    result["evidence_source"] = "narration_only"
    result["visual_role"] = "presenter_or_interview"
    result["usable_demo_after_relabel"] = "no"
    result["proposed_norm_supported"] = "no"
    result["rejection_reason"] = "presentation_only"
    assert parse_json(json.dumps(result), "v8") == result


def test_v8_repairs_out_of_schema_negative_aliases_without_creating_positive():
    result = valid_v8()
    for key in (
        "on_screen_social_event",
        "actor_performs_target_behavior",
        "affected_party_or_shared_context_same_event",
        "behavior_socially_evaluable_from_clip",
        "event_temporally_localized",
        "informal_social_conduct_not_formal_procedure",
        "usable_demo_after_relabel",
        "proposed_norm_supported",
    ):
        result[key] = "no"
    result["evidence_source"] = "metadata_only"
    result["visual_role"] = "presentation_only"
    result["rejection_reason"] = "metadata_only"
    parsed = parse_json(json.dumps(result), "v8")
    assert parsed["usable_demo_after_relabel"] == "no"
    assert parsed["visual_role"] == "presenter_or_interview"
    assert parsed["rejection_reason"] == "ambiguous_or_metadata_dependent"


def test_v8_repairs_model_surface_variants_fail_closed():
    result = valid_v8()
    for key in (
        "on_screen_social_event",
        "actor_performs_target_behavior",
        "affected_party_or_shared_context_same_event",
        "behavior_socially_evaluable_from_clip",
        "event_temporally_localized",
        "informal_social_conduct_not_formal_procedure",
        "usable_demo_after_relabel",
        "proposed_norm_supported",
    ):
        result[key] = "no"
    result["evidence_source"] = "narration_only"
    result["visual_role"] = "title_card"
    result["rejection_reason"] = "narration_only"
    parsed = parse_json(json.dumps(result), "v8")
    assert parsed["visual_role"] == "ambiguous_story_excerpt"
    assert parsed["rejection_reason"] == "presentation_only"
    assert parsed["usable_demo_after_relabel"] == "no"


def test_v8_repairs_exact_procedural_key_alias():
    result = valid_v8()
    result["informal_social_conduct_not_formal_procedural"] = result.pop(
        "informal_social_conduct_not_formal_procedure"
    )
    parsed = parse_json(json.dumps(result), "v8")
    assert parsed["informal_social_conduct_not_formal_procedure"] == "yes"
    assert (
        parsed["informal_social_conduct_key_validation_repair"]
        == "procedural_to_procedure_exact_key_alias"
    )
