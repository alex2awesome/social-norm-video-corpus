import json

from scripts.score_instructional_v9_alignment import (
    index_visual_records,
    parse_json,
    strict_visual_event,
)


def visual_record(observable: str = "yes") -> dict:
    return {
        "item_id": "instructional:test:0",
        "model": "test-vlm",
        "result": {
            "observable_event": observable,
            "event_start_percent": 10 if observable == "yes" else -1,
            "event_end_percent": 80 if observable == "yes" else -1,
            "literal_actor": "a customer" if observable == "yes" else "none",
            "literal_action_or_situated_utterance": (
                "asks a waiter for water" if observable == "yes" else "none"
            ),
            "literal_affected_party_or_shared_setting": (
                "the waiter at a table" if observable == "yes" else "none"
            ),
            "same_event_actor_action_target": "yes" if observable == "yes" else "no",
            "event_temporally_localized": "yes" if observable == "yes" else "no",
            "evidence_source": (
                "situated_dialogue_or_subtitles" if observable == "yes" else "none"
            ),
            "scene_role": "demonstrated_event" if observable == "yes" else "none",
            "demonstration_kind": (
                "explicit_social_etiquette" if observable == "yes" else "none"
            ),
        },
    }


def aligned_result() -> dict:
    return {
        "visual_demo_status": "exact",
        "proposed_norm_relation": "exact",
        "proposed_polarity_relation": "matches_event",
        "visible_event_polarity": "correct",
        "social_norm_type": "explicit_social_etiquette",
        "exact_weak_label_usable": "yes",
        "usable_after_relabel": "yes",
        "normalized_visible_behavior": "asking a waiter for a drink preference",
        "normalized_visible_polarity": "correct",
        "mismatch_reason": "none",
        "alignment_evidence": "The proposed table etiquette is the recorded request.",
    }


def test_strict_visual_event_accepts_explicit_etiquette():
    assert strict_visual_event(visual_record())


def test_v9b_accepts_consistent_exact_alignment():
    expected = aligned_result()
    assert parse_json(json.dumps(expected), visual_record()) == expected


def test_v9b_cannot_invent_a_demo_when_blind_visual_pass_found_none():
    result = aligned_result()
    parsed = parse_json(json.dumps(result), visual_record("no"))
    assert parsed["exact_weak_label_usable"] == "uncertain"
    assert parsed["usable_after_relabel"] == "uncertain"
    assert parsed["visual_demo_status"] == "uncertain"


def test_v9b_relabel_requires_a_normalized_visible_behavior():
    result = aligned_result()
    result["visual_demo_status"] = "usable_after_relabel"
    result["proposed_norm_relation"] = "mismatch"
    result["exact_weak_label_usable"] = "no"
    result["normalized_visible_behavior"] = "none"
    parsed = parse_json(json.dumps(result), visual_record())
    assert parsed["usable_after_relabel"] == "uncertain"
    assert parsed["visual_demo_status"] == "uncertain"


def test_v9b_does_not_promote_motor_skill_as_social_norm():
    result = aligned_result()
    result["visual_demo_status"] = "usable_after_relabel"
    result["proposed_norm_relation"] = "mismatch"
    result["social_norm_type"] = "motor_game_or_technical"
    result["exact_weak_label_usable"] = "no"
    parsed = parse_json(json.dumps(result), visual_record())
    assert parsed["usable_after_relabel"] == "uncertain"


def test_v9b_repairs_observed_schema_surface_variants_without_promotion():
    result = aligned_result()
    result["visual_demo_status"] = "usable_after_relabel"
    result["proposed_norm_relation"] = "broader_but_supported"
    result["proposed_polarity_relation"] = "polarity_mismatch"
    result["exact_weak_label_usable"] = "no"
    result["mismatch_reason"] = "broader_but_supported"
    parsed = parse_json(json.dumps(result), visual_record())
    assert parsed["proposed_norm_relation"] == "broader_but_supported"
    assert parsed["proposed_polarity_relation"] == "opposes_event"
    assert parsed["mismatch_reason"] == "none"
    assert parsed["exact_weak_label_usable"] == "no"


def test_storyboard_v9a_record_is_a_valid_frozen_visual_input():
    record = visual_record()
    record["rubric"] = "v9a_storyboard"
    assert index_visual_records([record])[record["item_id"]] == record


def test_v9b_repairs_32b_abstract_label_relation_fail_closed():
    result = aligned_result()
    result["visual_demo_status"] = "no_visual_demo"
    result["proposed_norm_relation"] = "abstract_label"
    result["social_norm_type"] = "abstract_or_off_topic"
    result["exact_weak_label_usable"] = "no"
    result["usable_after_relabel"] = "no"
    result["normalized_visible_behavior"] = "none"
    result["normalized_visible_polarity"] = "none"
    result["mismatch_reason"] = "abstract_label"
    parsed = parse_json(json.dumps(result), visual_record())
    assert parsed["proposed_norm_relation"] == "mismatch"
    assert parsed["exact_weak_label_usable"] == "no"


def test_out_of_schema_semantic_enum_forces_all_selection_gates_uncertain():
    result = aligned_result()
    result["social_norm_type"] = "health_hygiene"
    parsed = parse_json(json.dumps(result), visual_record())
    assert parsed["social_norm_type"] == "uncertain"
    assert parsed["exact_weak_label_usable"] == "uncertain"
    assert parsed["usable_after_relabel"] == "uncertain"
    assert parsed["visual_demo_status"] == "uncertain"
    assert parsed["out_of_schema_validation_repair"]["fields"] == [
        "social_norm_type"
    ]
