from scripts.export_commentary_temporal_baseline_manifest import build_rows


def test_build_rows_exports_exact_window_and_frozen_gold(tmp_path):
    proxy = tmp_path / "candidate-1.mp4"
    proxy.write_bytes(b"proxy")
    rows = build_rows(
        [
            {
                "audit_index": 1,
                "start_sec": 2.5,
                "end_sec": 6.0,
                "manual_gold": "positive",
            }
        ],
        [
            {
                "audit_index": 1,
                "candidate_id": "candidate-1",
                "uid": "source-1",
            }
        ],
        tmp_path,
    )
    assert rows == [
        {
            "audit_index": 1,
            "item_id": "candidate-1",
            "uid": "source-1",
            "pillar": "commentary",
            "source_clip": str(proxy.resolve()),
            "media_start_sec": 2.5,
            "media_end_sec": 6.0,
            "gold_scene_visible": True,
            "gold_social_scene_visible": True,
            "gold_label_matched_visible": True,
            "gold_usable": True,
        }
    ]
