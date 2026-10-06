import json

import pytest

from pathlib import Path

from scripts.run_instructional_v22_action_demo_vlm import parse_v22, rebase_sheet_paths


def valid() -> dict:
    return {
        "visual_role": "enacted_social_scenario",
        "connected_episode_visible": "yes",
        "concrete_performed_behavior_visible": "yes",
        "socially_interpretable_from_pixels": "yes",
        "only_explanation_report_or_broll": "no",
        "literal_visible_behavior": "one child throws an object at a teacher",
        "demo_pass": "yes",
        "evidence": "An animated child throws an object toward a teacher.",
    }


def test_parse_v22_accepts_consistent_demo() -> None:
    assert parse_v22(json.dumps(valid()))["demo_pass"] == "yes"


def test_parse_v22_repairs_inconsistent_positive() -> None:
    value = valid()
    value["visual_role"] = "talking_head_or_interview"
    parsed = parse_v22(json.dumps(value))
    assert parsed["demo_pass"] == "uncertain"
    assert parsed["consistency_repair"] == "inconsistent_positive_to_uncertain"


def test_parse_v22_rejects_schema_drift() -> None:
    value = valid()
    value["extra"] = True
    with pytest.raises(ValueError, match="schema mismatch"):
        parse_v22(json.dumps(value))


def test_rebase_sheet_paths_preserves_hash_and_uses_basename() -> None:
    rows = [{"sheet_path": "/old/host/000001.jpg", "sheet_sha256": "abc"}]
    assert rebase_sheet_paths(rows, Path("/new/host")) == [
        {"sheet_path": "/new/host/000001.jpg", "sheet_sha256": "abc"}
    ]
