import json

import pytest

from scripts.run_open_vlm_scene_benchmark import parse_json


def valid():
    return {
        "action_visible": "yes",
        "action_voluntary": "yes",
        "expectation_kind": "interpersonal_treatment",
        "reaction_visible_or_audibly_grounded": "yes",
        "reaction_source_role": "bystander",
        "reaction_content": "targeted_objection",
        "action_established_before_reaction": "yes",
        "reaction_targets_action": "yes",
        "authenticity": "organic",
        "proposed_label_relation": "repairable",
        "pre_reaction_demo_quality": "clear_audiovisual",
        "action_end_percent": 40,
        "reaction_start_percent": 50,
        "action_evidence": "A man approaches and threatens a rider.",
        "reaction_evidence": "An on-scene woman tells him to stop.",
        "evidence": "The objection follows and targets the confrontation.",
    }


def test_v7_atomic_response_and_bounds_parse():
    assert parse_json(json.dumps(valid()), "v7") == valid()


def test_v7_rejects_reversed_action_reaction_bounds():
    row = valid()
    row["action_end_percent"] = 60
    row["reaction_start_percent"] = 50
    with pytest.raises(ValueError, match="must not follow"):
        parse_json(json.dumps(row), "v7")


def test_v7_allows_unknown_bounds_to_fail_closed_later():
    row = valid()
    row["action_end_percent"] = -1
    row["reaction_start_percent"] = -1
    assert parse_json(json.dumps(row), "v7") == row


def test_v7_rejects_out_of_contract_enums():
    row = valid()
    row["reaction_source_role"] = "someone"
    with pytest.raises(ValueError, match="invalid v7 reaction_source_role"):
        parse_json(json.dumps(row), "v7")


def test_v7b_uses_same_atomic_schema_and_requires_paired_bounds():
    assert parse_json(json.dumps(valid()), "v7b") == valid()
    row = valid()
    row["action_end_percent"] = -1
    with pytest.raises(ValueError, match="both be -1"):
        parse_json(json.dumps(row), "v7b")


def test_v7c_repairs_reversed_bounds_to_unknown_fail_closed():
    row = valid()
    row["action_end_percent"] = 60
    row["reaction_start_percent"] = 50
    parsed = parse_json(json.dumps(row), "v7c")
    assert parsed["action_end_percent"] == -1
    assert parsed["reaction_start_percent"] == -1
    assert (
        parsed["temporal_bounds_validation_repair"]
        == "reverse_or_overlap_fail_closed"
    )


def test_v7c_leaves_valid_bounds_unchanged():
    row = valid()
    assert parse_json(json.dumps(row), "v7c") == row


def test_v7c_repairs_unpaired_bound_to_unknown_fail_closed():
    row = valid()
    row["action_end_percent"] = -1
    parsed = parse_json(json.dumps(row), "v7c")
    assert parsed["action_end_percent"] == -1
    assert parsed["reaction_start_percent"] == -1
    assert (
        parsed["temporal_bounds_validation_repair"]
        == "unpaired_bound_fail_closed"
    )


def test_v7c_repairs_out_of_range_bounds_to_unknown_fail_closed():
    row = valid()
    row["action_end_percent"] = 140
    row["reaction_start_percent"] = 160
    parsed = parse_json(json.dumps(row), "v7c")
    assert parsed["action_end_percent"] == -1
    assert parsed["reaction_start_percent"] == -1
    assert (
        parsed["temporal_bounds_validation_repair"]
        == "out_of_range_fail_closed"
    )


def test_v7c_repairs_no_authenticity_to_uncertain_fail_closed():
    row = valid()
    row["authenticity"] = "none"
    parsed = parse_json(json.dumps(row), "v7c")
    assert parsed["authenticity"] == "uncertain"
    assert (
        parsed["authenticity_validation_repair"]
        == "none_to_uncertain_fail_closed"
    )


def test_v7c_repairs_unknown_enum_to_uncertain_fail_closed():
    row = valid()
    row["reaction_content"] = "reassurance"
    parsed = parse_json(json.dumps(row), "v7c")
    assert parsed["reaction_content"] == "uncertain"
    assert (
        parsed["reaction_content_validation_repair"]
        == "out_of_schema_reassurance_to_uncertain_fail_closed"
    )


def test_v7c_recovers_only_truncated_final_evidence_tail():
    row = valid()
    complete = json.dumps(row)
    truncated = complete.split(', "evidence":', 1)[0] + ', "evidence": "loops'

    parsed = parse_json(truncated, "v7c")

    for key, value in row.items():
        if key != "evidence":
            assert parsed[key] == value
    assert parsed["evidence"] == "truncated after complete atomic fields"
    assert (
        parsed["evidence_validation_repair"]
        == "truncated_free_text_tail_replaced"
    )
