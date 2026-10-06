import json
from pathlib import Path

from scripts.build_instructional_v9_storyboard_manifest import build
from scripts.run_open_vlm_v9a_storyboards import STORYBOARD_INSTRUCTION


def test_storyboard_prompt_is_label_blind():
    lowered = STORYBOARD_INSTRUCTION.lower()
    assert "no title" in lowered
    assert "norm" in lowered
    assert "do not infer" in lowered


def test_builder_emits_only_label_blind_identifiers_and_media(tmp_path: Path):
    selection = tmp_path / "selection.jsonl"
    dense = tmp_path / "dense.jsonl"
    selection.write_text(
        json.dumps(
            {
                "audit_index": 0,
                "item_id": "instructional:test:0",
                "uid": "test",
                "pillar": "instructional",
                "norm": "secret norm",
                "polarity": "violation",
            }
        )
        + "\n"
    )
    dense.write_text(
        json.dumps(
            {
                "audit_index": 0,
                "item_id": "candidate:00",
                "uid": "test",
                "sheet_path": "old/place/00.jpg",
                "sheet_sha256": "abc",
                "media": {"frame_count": 36, "fps": 3.0},
            }
        )
        + "\n"
    )
    rows = build(selection, dense, Path("/frozen/sheets"))
    assert rows[0]["sheet_path"] == "/frozen/sheets/00.jpg"
    assert "norm" not in rows[0]
    assert "polarity" not in rows[0]
