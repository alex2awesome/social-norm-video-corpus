import pytest

from scripts.export_fresh_scene_query_audit import (
    commentary_interval,
    demos_by_clip,
    numeric_suffix,
    sample_timestamps,
)


def test_commentary_interval_includes_preceding_context_and_clamps():
    assert commentary_interval(5.0, 6.0, 20.0) == (0.0, 8.0)
    assert commentary_interval(18.0, 19.0, 20.0) == (6.0, 20.0)


def test_commentary_interval_rejects_empty_media():
    with pytest.raises(ValueError, match="empty"):
        commentary_interval(0.0, 0.0, 0.0)


def test_sample_timestamps_are_centered_and_ordered():
    values = sample_timestamps(10.0, 22.0, 3)
    assert values == [12.0, 16.0, 20.0]


def test_numeric_suffix():
    from pathlib import Path

    assert numeric_suffix(Path("demo_12.mp4")) == 12


def test_missing_clip_name_preserves_original_demo_index():
    rows = [{"norm": "first", "clip": "demo_0.mp4"}, {"norm": "invalid"}]
    assert demos_by_clip(rows)["demo_1.mp4"]["norm"] == "invalid"
