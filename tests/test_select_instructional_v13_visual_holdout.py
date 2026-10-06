import json
from pathlib import Path

from scripts.select_instructional_v13_visual_holdout import (
    is_candidate,
    select,
    unique_index,
)


def _visual(item_id: str, strict: bool) -> dict:
    return {
        "item_id": item_id,
        "error": None,
        "result": {
            "observable_event": "yes" if strict else "no",
            "event_temporally_localized": "yes" if strict else "no",
            "same_event_actor_action_target": "yes" if strict else "no",
            "aftermath_only": "no",
            "montage_only": "no",
            "presentation_only": "no",
            "metadata_needed_to_name_action": "no",
            "demonstration_kind": (
                "interpersonal_conduct" if strict else "none"
            ),
            "evidence_source": "physical_action" if strict else "none",
            "scene_role": "demonstrated_event" if strict else "none",
            "literal_actor": "actor" if strict else "none",
            "literal_action_or_situated_utterance": (
                "performs action" if strict else "none"
            ),
            "literal_affected_party_or_shared_setting": (
                "affected party" if strict else "none"
            ),
            "event_start_percent": 10 if strict else -1,
            "event_end_percent": 70 if strict else -1,
        },
    }


def _alignment(item_id: str, usable: bool) -> dict:
    return {
        "item_id": item_id,
        "error": None,
        "result": {"usable_after_relabel": "yes" if usable else "no"},
    }


def test_is_candidate_requires_all_frozen_conditions() -> None:
    source = {
        "item_id": "i0",
        "source_platform": "youtube",
        "polarity": "violation",
    }
    assert is_candidate(source, _visual("i0", True), _alignment("i0", True))
    assert not is_candidate(source, _visual("i0", False), _alignment("i0", True))
    assert not is_candidate(source, _visual("i0", True), _alignment("i0", False))
    assert not is_candidate(
        {**source, "source_platform": "dailymotion"},
        _visual("i0", True),
        _alignment("i0", True),
    )


def test_select_excludes_prior_uids_and_blinds_semantics() -> None:
    sources = [
        {
            "item_id": f"i{i}",
            "uid": f"u{i}",
            "source_platform": "youtube",
            "polarity": "violation",
            "norm": "hidden norm",
        }
        for i in range(4)
    ]
    storyboards = {
        f"i{i}": {
            "item_id": f"i{i}",
            "sheet_path": f"storyboards/{i}.jpg",
            "sheet_sha256": str(i) * 64,
        }
        for i in range(4)
    }
    visuals = {
        "i0": _visual("i0", True),
        "i1": _visual("i1", True),
        "i2": _visual("i2", True),
        "i3": _visual("i3", False),
    }
    alignments = {
        "i0": _alignment("i0", True),
        "i1": _alignment("i1", True),
        "i2": _alignment("i2", False),
        "i3": _alignment("i3", False),
    }
    semantic, blind, summary = select(
        sources, storyboards, visuals, alignments, {"u1"}
    )
    assert summary["candidate_rows_available"] == 1
    assert any(row["band"] == "primary_v13_candidate" for row in semantic)
    assert all("norm" not in row for row in blind)
    assert all(row["uid"] != "u1" for row in semantic)


def test_unique_index_accepts_storyboard_rows_without_result(tmp_path: Path) -> None:
    path = tmp_path / "storyboards.jsonl"
    path.write_text(json.dumps({"item_id": "i0", "sheet_path": "0.jpg"}) + "\n")
    assert unique_index(path)["i0"]["sheet_path"] == "0.jpg"
