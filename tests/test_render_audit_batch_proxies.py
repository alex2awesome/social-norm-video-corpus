from pathlib import Path

import pytest

from scripts.render_audit_batch_proxies import (
    file_sha256,
    proxy_name,
    sampling_fps,
)


def test_proxy_name_is_deterministic_and_safe():
    name = proxy_name("instructional:dailymotion__x/1:2")

    assert "/" not in name
    assert name.endswith(".mp4")
    assert name == proxy_name("instructional:dailymotion__x/1:2")
    assert name != proxy_name("instructional:dailymotion__x/1:3")


def test_file_sha256(tmp_path: Path):
    path = tmp_path / "proxy.mp4"
    path.write_bytes(b"proxy")

    assert file_sha256(path) == (
        "1241936d4dd3aad68fe7bfbdfe854b935926bc678fc72377e15166078916227a"
    )


def test_sampling_fps_caps_long_clips_without_oversampling_short_clips():
    assert sampling_fps(10, 3, 96) == 3
    assert sampling_fps(120, 3, 96) == pytest.approx(0.8)

    with pytest.raises(ValueError):
        sampling_fps(0, 3, 96)
