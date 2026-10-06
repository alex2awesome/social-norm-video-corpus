import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_open_vlm_v10_storyboards.py"
SPEC = importlib.util.spec_from_file_location("open_vlm_v10", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_result():
    return {
        "performed_social_behavior": "yes",
        "actor_action_target_same_event": "yes",
        "target_behavior_performed_not_described": "yes",
        "affected_party_or_shared_setting_present": "yes",
        "socially_evaluable_without_metadata": "yes",
        "social_scope": "tacit_interpersonal",
        "scene_role": "situated_scene",
        "social_response_or_consequence_present": "yes",
        "literal_actor": "one coworker",
        "literal_action_or_situated_utterance": "insults another coworker",
        "literal_affected_party_or_shared_setting": "the other coworker",
        "demo_usable": "yes",
        "rejection_reason": "none",
        "evidence": "One coworker insults another, who visibly responds.",
    }


def test_valid_positive_parses():
    assert MODULE.parse_v10a(json.dumps(valid_result()))["demo_usable"] == "yes"


def test_inconsistent_positive_is_failed_closed():
    row = valid_result()
    row["target_behavior_performed_not_described"] = "no"
    parsed = MODULE.parse_v10a(json.dumps(row))
    assert parsed["demo_usable"] == "uncertain"
    assert parsed["consistency_repair"] == "inconsistent_positive_to_uncertain"


def test_extra_schema_field_is_rejected():
    row = valid_result()
    row["invented"] = "yes"
    with pytest.raises(ValueError, match="schema mismatch"):
        MODULE.parse_v10a(json.dumps(row))


def test_unknown_enum_is_recorded_and_failed_closed():
    row = valid_result()
    row["rejection_reason"] = "presenter_sample_or_advice"
    parsed = MODULE.parse_v10a(json.dumps(row))
    assert parsed["rejection_reason"] == "uncertain"
    assert parsed["rejection_reason_raw"] == "presenter_sample_or_advice"
    assert parsed["demo_usable"] == "uncertain"
    assert parsed["enum_repairs"] == ["rejection_reason"]


def test_opaque_manifest_identity_does_not_require_or_invent_uid():
    row = {
        "item_id": "commentary-title-pop-0007",
        "candidate_id": "commentary-title-pop-0007",
        "audit_index": 7,
        "pillar": "commentary",
        "sheet_sha256": "abc",
        "frame_count": 96,
    }
    record = MODULE.base_record(row, "gemma-3-27b-it")
    assert record["candidate_id"] == "commentary-title-pop-0007"
    assert record["pillar"] == "commentary"
    assert "uid" not in record


def test_source_manifest_identity_keeps_uid():
    row = {
        "item_id": "instructional:youtube__abc:0",
        "uid": "youtube__abc",
        "pillar": "instructional",
        "sheet_sha256": "def",
        "frame_count": 36,
    }
    record = MODULE.base_record(row, "glm")
    assert record["uid"] == "youtube__abc"
    assert "candidate_id" not in record
