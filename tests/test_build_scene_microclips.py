import json
from types import SimpleNamespace

import scripts.build_scene_microclips as build_scene_microclips
from scripts.build_scene_microclips import (
    probe_duration,
    probe_video_seek_offset,
    probe_video_start,
    windows,
)


def test_probe_duration_prefers_video_stream_duration_over_container_end_time():
    assert probe_duration(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "start_time": "6.333008",
                    "duration": "9.333333",
                }
            ],
            "format": {"start_time": "6.333008", "duration": "15.667000"},
        }
    ) == 9.333333


def test_probe_video_start_preserves_positive_stream_offset_for_seeking():
    assert probe_video_start(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "start_time": "8.333008",
                    "duration": "2.333333",
                },
                {"codec_type": "audio", "start_time": "0.0", "duration": "10.56"},
            ],
            "format": {"start_time": "0.0", "duration": "10.666341"},
        }
    ) == 8.333008


def test_probe_video_seek_offset_is_relative_to_container_start():
    assert probe_video_seek_offset(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "start_time": "8.333008",
                    "duration": "2.333333",
                },
                {"codec_type": "audio", "start_time": "0.0", "duration": "10.56"},
            ],
            "format": {"start_time": "0.0", "duration": "10.666341"},
        }
    ) == 8.333008
    assert probe_video_seek_offset(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "start_time": "8.0",
                    "duration": "5.0",
                }
            ],
            "format": {"start_time": "8.0", "duration": "13.0"},
        }
    ) == 0.0


def test_video_timing_requests_stream_start_time(monkeypatch, tmp_path):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "start_time": "8.333008",
                            "duration": "2.333333",
                        }
                    ],
                    "format": {"start_time": "0.0", "duration": "10.666341"},
                }
            )
        )

    monkeypatch.setattr(build_scene_microclips.subprocess, "run", fake_run)
    assert build_scene_microclips.video_timing(tmp_path / "clip.mp4") == (
        8.333008,
        2.333333,
    )
    show_entries = observed["command"][
        observed["command"].index("-show_entries") + 1
    ]
    assert "stream=codec_type,start_time,duration" in show_entries


def test_probe_duration_fallback_subtracts_positive_start_time():
    assert probe_duration(
        {
            "streams": [{"codec_type": "video", "duration": "N/A"}],
            "format": {"start_time": "7.5", "duration": "12.5"},
        }
    ) == 5.0


def test_windows_do_not_create_zero_length_tail():
    assert windows(5.0, 6.0, 3.0) == [(0.0, 5.0)]
    assert windows(9.0, 6.0, 3.0) == [(0.0, 6.0), (3.0, 9.0)]
