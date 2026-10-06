import json

from scripts.export_instructional_demo_population_v1 import export_population


def test_export_includes_all_materialized_polarities_and_skips_missing(tmp_path):
    root = tmp_path / "instructional"
    source = root / "dailymotion__a"
    source.mkdir(parents=True)
    (source / "demo_0.mp4").write_bytes(b"video")
    (source / "demo_1.mp4").write_bytes(b"video")
    (source / "metadata.json").write_text(json.dumps({
        "video_id": "dailymotion__a",
        "title": "Etiquette demonstration",
        "category": "instr_etiquette",
        "demos": [
            {"clip": "demo_0.mp4", "polarity": "violation", "norm": "be polite"},
            {"clip": "demo_1.mp4", "polarity": "correct", "norm": "be polite"},
            {"clip": "missing.mp4", "polarity": "explanation", "norm": "be polite"},
        ],
    }))
    rows, summary = export_population(root)
    assert [row["polarity"] for row in rows] == ["violation", "correct"]
    assert all(row["source_clip"].endswith(".mp4") for row in rows)
    assert summary["demos"] == 2
    assert summary["sources"] == 1
    assert summary["missing_clip_records"] == 1
    assert summary["automatic_acceptance"] is False


def test_metadata_directory_name_is_uid_fallback(tmp_path):
    source = tmp_path / "instructional" / "youtube__b"
    source.mkdir(parents=True)
    (source / "d.mp4").write_bytes(b"x")
    (source / "metadata.json").write_text(json.dumps({
        "demos": [{"clip": "d.mp4", "polarity": "explanation"}],
        "provenance": {"category": "instr_family"},
    }))
    rows, _ = export_population(tmp_path / "instructional")
    assert rows[0]["uid"] == "youtube__b"
    assert rows[0]["category"] == "instr_family"
