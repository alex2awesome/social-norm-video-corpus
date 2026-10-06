import json

from pathlib import Path

import pytest

from scripts.run_commentary_title_event_vlm import (
    join_manifests,
    parse_result,
    resolve_sheet,
)


def valid_payload() -> dict:
    return {
        "candidate_event": "uncertain",
        "title_action_visible": "uncertain",
        "title_actor_visible": "yes",
        "title_target_or_property_visible": "yes",
        "actor_action_target_same_event": "uncertain",
        "event_kind": "property_taking",
        "candidate_start_sec": 10,
        "candidate_end_sec": 14,
        "followup": "motion_and_crop",
        "literal_candidate": "A person bends beside a parcel.",
        "title_alignment": "unresolved",
        "evidence": "Frames at 10-14 seconds show handling but not removal.",
    }


def test_join_manifests_adds_only_title_semantics() -> None:
    rows = join_manifests(
        [{"audit_index": 1, "candidate_id": "c1", "item_id": "i1"}],
        [{"candidate_id": "c1", "norm": "Person takes parcel", "uid": "u1"}],
    )
    assert rows == [
        {
            "audit_index": 1,
            "candidate_id": "c1",
            "item_id": "i1",
            "title": "Person takes parcel",
        }
    ]


def test_join_manifests_falls_back_to_item_id_and_restores_candidate_id() -> None:
    rows = join_manifests(
        [{"item_id": "i1", "storyboard_index": 7, "sheet_path": "storyboards/1.jpg"}],
        [{
            "audit_index": 3,
            "candidate_id": "c1",
            "item_id": "i1",
            "norm": "Person takes parcel",
        }],
    )
    assert rows[0]["candidate_id"] == "c1"
    assert rows[0]["audit_index"] == 3
    assert rows[0]["title"] == "Person takes parcel"


def test_join_manifests_uses_storyboard_index_for_legacy_semantics() -> None:
    rows = join_manifests(
        [{"item_id": "i1", "storyboard_index": 7}],
        [{"candidate_id": "c1", "item_id": "i1", "norm": "event"}],
    )
    assert rows[0]["audit_index"] == 7


def test_join_manifests_rejects_duplicate_item_ids() -> None:
    with pytest.raises(ValueError, match="duplicate semantic item_id"):
        join_manifests(
            [{"item_id": "i1"}],
            [
                {"item_id": "i1", "norm": "one"},
                {"item_id": "i1", "norm": "two"},
            ],
        )


def test_resolve_sheet_rebases_sealed_basename() -> None:
    row = {"sheet_path": "/remote/storyboards/000007.jpg"}
    assert resolve_sheet(row, Path("/manifest"), Path("/copied")) == Path(
        "/copied/000007.jpg"
    )


def test_resolve_sheet_preserves_explicit_decode_failure() -> None:
    assert resolve_sheet({"sheet_path": None}, Path("/manifest"), None) is None


def test_parse_result_accepts_bounded_uncertain() -> None:
    result = parse_result(json.dumps(valid_payload()))
    assert result["candidate_event"] == "uncertain"
    assert result["candidate_start_sec"] == 10


def test_parse_result_repairs_unsupported_yes() -> None:
    payload = valid_payload()
    payload["candidate_event"] = "yes"
    result = parse_result(json.dumps(payload))
    assert result["candidate_event"] == "uncertain"
    assert result["consistency_repair"] == "unsupported_yes_to_uncertain"


def test_parse_result_clears_bounds_for_no_event() -> None:
    payload = valid_payload()
    payload["candidate_event"] = "no"
    result = parse_result(json.dumps(payload))
    assert result["candidate_start_sec"] is None
    assert result["candidate_end_sec"] is None
    assert result["bounds_repair"] == "no_event_bounds_cleared"
