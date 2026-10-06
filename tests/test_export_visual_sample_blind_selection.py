from pathlib import Path

from scripts.export_visual_sample_blind_selection import convert


def test_convert_hides_semantics_from_blind_rows(tmp_path: Path):
    manifest = {
        "visual_samples": [
            {
                "audit_index": 0,
                "pillar": "commentary",
                "uid": "u",
                "clip_path": "data/discussion_video/u.mp4",
                "norm": "respect",
                "statement": "That was rude",
            }
        ]
    }
    sealed, blind = convert(manifest, tmp_path)
    assert sealed[0]["norm"] == "respect"
    assert "norm" not in blind[0]
    assert blind[0]["clip"] == str(tmp_path / "data/discussion_video/u.mp4")
