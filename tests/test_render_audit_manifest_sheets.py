from scripts.render_audit_manifest_sheets import normalized_frames


def test_normalized_frames_accept_source_timestamp():
    frames = normalized_frames(
        [{"frame_index": 0, "source_timestamp": 12.5, "clip_timestamp": 2.5}]
    )
    assert frames[0]["timestamp"] == 12.5


def test_normalized_frames_preserve_generic_timestamp():
    frames = normalized_frames(
        [{"frame_index": 0, "timestamp": 3.0, "source_timestamp": 12.5}]
    )
    assert frames[0]["timestamp"] == 3.0
