import json
from types import SimpleNamespace

import src.media_integrity as media_integrity
from src.media_integrity import parse_media_timing, probe_media_timing, timing_flags


def test_timing_detects_video_delayed_behind_audio():
    timing = parse_media_timing(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "start_time": "8.333008",
                    "duration": "2.333333",
                },
                {
                    "codec_type": "audio",
                    "start_time": "0.0",
                    "duration": "10.56",
                },
            ],
            "format": {"start_time": "0.0", "duration": "10.666341"},
        }
    )
    assert timing["leading_video_gap_sec"] == 8.333008
    assert timing["video_coverage_ratio"] < 0.25
    assert timing_flags(timing) == [
        "delayed_video_start",
        "short_video_coverage",
    ]


def test_timing_handles_video_only_nonzero_container_timestamp():
    timing = parse_media_timing(
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
    )
    assert timing["timeline_duration_sec"] == 5.0
    assert timing["leading_video_gap_sec"] == 0.0
    assert timing["video_coverage_ratio"] == 1.0
    assert timing_flags(timing) == []


def test_timing_uses_stream_timeline_for_positive_source_start():
    timing = parse_media_timing(
        {
            "streams": [
                {
                    "codec_type": "audio",
                    "start_time": "9.976789",
                    "duration": "98.916989",
                },
                {
                    "codec_type": "video",
                    "start_time": "10.0",
                    "duration": "99.066667",
                },
            ],
            "format": {"start_time": "9.976789", "duration": "99.089878"},
        }
    )
    assert timing["leading_video_gap_sec"] < 0.03
    assert timing["video_coverage_ratio"] > 0.99
    assert timing_flags(timing) == []


def test_probe_uses_environment_sibling_when_ffprobe_not_on_path(
    monkeypatch, tmp_path
):
    runtime = tmp_path / "bin" / "python"
    runtime.parent.mkdir()
    runtime.write_text("")
    ffprobe = runtime.with_name("ffprobe")
    ffprobe.write_text("")
    observed = {}

    monkeypatch.setattr(media_integrity.sys, "executable", str(runtime))
    monkeypatch.setattr(media_integrity.shutil, "which", lambda _name: None)

    def fake_run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "start_time": "0",
                            "duration": "2",
                        }
                    ],
                    "format": {"start_time": "0", "duration": "2"},
                }
            )
        )

    monkeypatch.setattr(media_integrity.subprocess, "run", fake_run)
    probe_media_timing(tmp_path / "clip.mp4")
    assert observed["command"][0] == str(ffprobe)
