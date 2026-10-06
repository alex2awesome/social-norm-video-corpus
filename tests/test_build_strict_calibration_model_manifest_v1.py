from pathlib import Path

from scripts.build_strict_calibration_model_manifest_v1 import build_rows


def test_build_rows_aligns_witnessed_transcript_and_gold() -> None:
    selection = {
        "audit_index": 4,
        "item_id": "witnessed:u:0",
        "uid": "u",
        "pillar": "witnessed",
        "media_path": "data/hits/u/clip.mp4",
        "start_sec": 10.0,
        "end_sec": 20.0,
        "norm": "rudeness",
        "reaction_tag": "objection",
        "reaction_text": "stop that",
    }
    semantic = {
        "audit_index": 4,
        "item_id": "witnessed:u:0",
        "transcript_segments": [{"start": 12, "end": 13, "text": " stop "}],
        "semantic": {},
    }
    review = {
        "audit_index": 4,
        "pillar": "witnessed",
        "strict_decision": "reject",
        "salvage_route": "instructional_review",
        "social_norm_scene": "yes",
        "label_alignment": "yes",
    }
    row = build_rows(Path("/corpus"), [selection], [semantic], [review])[0]
    assert row["aligned_transcript"] == [{"start": 2.0, "end": 3.0, "text": "stop"}]
    assert row["gold_social_scene_visible"] is True
    assert row["gold_witnessed_strict"] is False
    assert row["explanation"] == "objection: stop that"


def test_build_rows_bounds_commentary_and_marks_repairable() -> None:
    selection = {
        "audit_index": 2,
        "item_id": "commentary:u:1",
        "uid": "u",
        "pillar": "commentary",
        "media_path": "data/discussion/u/source.mp4",
        "start_sec": 20.0,
        "end_sec": 22.0,
        "norm": "courtesy",
    }
    semantic = {
        "audit_index": 2,
        "item_id": "commentary:u:1",
        "transcript_segments": [{"start": 9, "end": 10, "text": "hello"}],
        "semantic": {"statement": {"quote": "be kind"}},
    }
    review = {
        "audit_index": 2,
        "pillar": "commentary",
        "strict_decision": "accept_with_repairs",
        "salvage_route": "commentary_visual_relabel",
        "social_norm_scene": "yes",
        "label_alignment": "uncertain",
    }
    row = build_rows(Path("/corpus"), [selection], [semantic], [review])[0]
    assert row["media_start_sec"] == 8.0
    assert row["media_end_sec"] == 34.0
    assert row["duration_hint"] == 26.0
    assert row["aligned_transcript"][0]["start"] == 1.0
    assert row["gold_usable"] is False
    assert row["gold_repairable"] is True
