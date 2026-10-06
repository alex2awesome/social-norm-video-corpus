from pathlib import Path
from types import SimpleNamespace

from src import clip_extract


def test_reencode_fallback_seeks_after_input(monkeypatch, tmp_path):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(clip_extract.subprocess, "run", fake_run)
    monkeypatch.setattr(clip_extract, "_duration_ok", lambda _path: True)
    assert clip_extract._cut_reencode(
        tmp_path / "source.mp4",
        tmp_path / "target.mp4",
        10.0,
        20.0,
    )
    command = observed["command"]
    assert command.index("-i") < command.index("-ss")


def test_duration_ok_rejects_delayed_video(monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 2048)
    monkeypatch.setattr(
        clip_extract,
        "probe_media_timing",
        lambda _path, timeout: {
            "has_video": True,
            "has_audio": True,
            "timeline_duration_sec": 10.0,
            "leading_video_gap_sec": 5.0,
            "video_coverage_ratio": 0.5,
        },
    )
    assert not clip_extract._duration_ok(clip)


def test_duration_ok_accepts_aligned_playable_video(monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 2048)
    monkeypatch.setattr(
        clip_extract,
        "probe_media_timing",
        lambda _path, timeout: {
            "has_video": True,
            "has_audio": True,
            "timeline_duration_sec": 10.0,
            "leading_video_gap_sec": 0.02,
            "video_coverage_ratio": 0.99,
        },
    )
    assert clip_extract._duration_ok(clip)
