import json
from pathlib import Path

from scripts.run_instructional_v15_structure_vlm import (
    normalize_manifest_row,
    parse_v15s,
)


def valid() -> dict:
    return {
        "visible_people_or_characters": "multiple",
        "interaction_structure": "specific_action_with_target",
        "causative_social_act_visible": "yes",
        "affected_party_in_depicted_event": "yes",
        "action_specificity": "specific",
        "target_is_person_or_social_group": "yes",
        "only_report_or_advice": "no",
        "literal_actor": "one child",
        "literal_action_or_utterance": "takes a toy",
        "literal_affected_party": "another child",
        "structural_demo_pass": "yes",
        "evidence": "One child takes a toy and the other child cries.",
    }


def test_parse_accepts_strict_structural_demo():
    assert parse_v15s(json.dumps(valid()))["structural_demo_pass"] == "yes"


def test_parse_repairs_inconsistent_positive():
    row = valid()
    row["action_specificity"] = "generic"
    parsed = parse_v15s(json.dumps(row))
    assert parsed["structural_demo_pass"] == "uncertain"
    assert parsed["structural_demo_pass_raw"] == "yes"


def test_parse_rejects_schema_drift():
    row = valid()
    row["extra"] = "bad"
    try:
        parse_v15s(json.dumps(row))
    except ValueError as exc:
        assert "schema mismatch" in str(exc)
    else:
        raise AssertionError("schema drift should fail")


def test_normalize_manifest_row_supports_opaque_blind_packet():
    row = normalize_manifest_row(
        {
            "candidate_id": "blind-7",
            "audit_index": 7,
            "sheet_path": "storyboards/7.jpg",
            "sheet_sha256": "abc",
        },
        Path("/audit"),
        "instructional",
    )
    assert row["item_id"] == "blind-7"
    assert row["uid"] == "blind-7"
    assert row["frame_count"] == 36
    assert row["pillar"] == "instructional"
    assert row["_manifest_dir"] == "/audit"
