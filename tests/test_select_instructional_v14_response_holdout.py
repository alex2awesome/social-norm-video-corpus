import json

from scripts.select_instructional_v14_response_holdout import select, stage


def visual(item_id: str, *, strict: bool = True) -> dict:
    result = {
        "observable_event": "yes",
        "event_start_percent": 10,
        "event_end_percent": 40,
        "literal_actor": "person",
        "literal_action_or_situated_utterance": "interrupts",
        "literal_affected_party_or_shared_setting": "other person",
        "same_event_actor_action_target": "yes",
        "event_temporally_localized": "yes",
        "evidence_source": "physical_action",
        "scene_role": "demonstrated_event",
        "demonstration_kind": "interpersonal_conduct",
    }
    if not strict:
        result["observable_event"] = "no"
    return {"item_id": item_id, "result": result}


def verifier(
    item_id: str, *, usable: str = "yes", response: str = "yes"
) -> dict:
    return {
        "item_id": item_id,
        "result": {
            "demo_usable": usable,
            "social_response_or_consequence_present": response,
        },
    }


def test_stage_requires_visual_demo_and_visible_response():
    assert stage(visual("a"), verifier("a")) == "candidate"
    assert stage(visual("a", strict=False), verifier("a")) == "v9a_reject"
    assert stage(visual("a"), verifier("a", usable="no")) == "v10a_reject"
    assert stage(visual("a"), verifier("a", response="no")) == "response_reject"


def test_selection_keeps_all_candidates_and_blinds_semantics():
    sources = [
        {"item_id": f"i{i}", "uid": f"u{i}", "norm": f"secret{i}"}
        for i in range(7)
    ]
    storyboards = {
        f"i{i}": {
            "item_id": f"i{i}",
            "sheet_path": f"/s/{i}.jpg",
            "sheet_sha256": str(i),
        }
        for i in range(7)
    }
    visuals = {f"i{i}": visual(f"i{i}") for i in range(7)}
    verifiers = {
        "i0": verifier("i0"),
        "i1": verifier("i1"),
        "i2": verifier("i2", response="no"),
        "i3": verifier("i3", response="no"),
        "i4": verifier("i4", usable="no"),
        "i5": verifier("i5", usable="no"),
        "i6": verifier("i6", usable="no"),
    }
    semantic, blind, summary = select(
        sources,
        storyboards,
        visuals,
        verifiers,
        controls_per_band=2,
    )

    assert summary["candidate_rows"] == 2
    assert summary["control_counts"] == {
        "response_reject": 2,
        "v10a_reject": 2,
    }
    assert len(semantic) == len(blind) == 6
    assert sum(row["band"] == "primary_v14_candidate" for row in semantic) == 2
    assert all("norm" not in row and "item_id" not in row for row in blind)
    assert all(set(row) == {
        "audit_index", "candidate_id", "sheet_path", "sheet_sha256"
    } for row in blind)


def test_duplicate_source_uid_is_rejected():
    source = [
        {"item_id": "i0", "uid": "u"},
        {"item_id": "i1", "uid": "u"},
    ]
    storyboards = {
        item: {"item_id": item, "sheet_path": f"{item}.jpg", "sheet_sha256": item}
        for item in ("i0", "i1")
    }
    visuals = {item: visual(item) for item in ("i0", "i1")}
    verifiers = {item: verifier(item) for item in ("i0", "i1")}
    try:
        select(source, storyboards, visuals, verifiers, controls_per_band=1)
    except ValueError as exc:
        assert "one item per UID" in str(exc)
    else:
        raise AssertionError("expected duplicate UID rejection")
