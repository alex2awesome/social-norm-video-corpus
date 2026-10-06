from scripts.build_commentary_hierarchical_window_manifest_v2 import (
    build_windows,
    validate_full_coverage,
)


def test_complete_tiling_covers_source_tail():
    windows = build_windows([{
        "uid": "u",
        "item_id": "commentary:u:0",
        "source_path": "/tmp/u.mp4",
        "duration_sec": 25.0,
        "title": "t",
        "action_label": "one person hits another",
        "anchors": [],
    }])
    validate_full_coverage(windows)
    assert windows[0]["window_start_sec"] == 0
    assert windows[-1]["window_end_sec"] == 25
    assert all(row["automatic_acceptance"] is False for row in windows)
