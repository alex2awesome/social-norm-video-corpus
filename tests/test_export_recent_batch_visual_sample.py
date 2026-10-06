from scripts.export_recent_batch_visual_sample import frame_times, numeric_anchors, stable_key


def test_commentary_times_center_on_first_grounded_anchor():
    assert frame_times(100.0, "commentary", [30.0]) == [26.0, 30.0, 34.0]


def test_non_commentary_times_cover_the_clip():
    assert frame_times(80.0, "witnessed", []) == [20.0, 40.0, 60.0]


def test_nested_metadata_anchors_are_discovered():
    assert numeric_anchors({"matches": [{"start": 3}, {"timestamp": 9.5}]}) == [3.0, 9.5]


def test_stable_key_is_deterministic():
    assert stable_key("uid") == stable_key("uid")
