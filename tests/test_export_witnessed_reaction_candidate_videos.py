from pathlib import Path

from scripts.export_witnessed_reaction_candidate_videos import (
    bounded_window,
    export,
    ffmpeg_argv,
)


def test_bounded_window_is_relative_and_clamped():
    row = {"media_start_sec": -2, "media_end_sec": 8, "candidate_start_sec": 3, "candidate_end_sec": 4}
    assert bounded_window(row) == (0.0, 8.0, 3.0, 4.0)


def test_bounded_window_clamps_to_source_duration():
    row = {"media_start_sec": 5, "media_end_sec": 12, "candidate_start_sec": 8, "candidate_end_sec": 11}
    assert bounded_window(row, 9.0) == (5.0, 9.0, 3.0, 4.0)


def test_ffmpeg_window_reencodes_audio_and_video():
    argv = ffmpeg_argv("/bin/ffmpeg", Path("in.mp4"), Path("out.mp4"), 1.5, 4.5)
    assert argv[argv.index("-ss") + 1] == "1.500"
    assert argv[argv.index("-t") + 1] == "3.000"
    assert "libx264" in argv
    assert "aac" in argv


def test_ffmpeg_window_can_standardize_model_frame_rate_and_width():
    argv = ffmpeg_argv(
        "ffmpeg", Path("in.mp4"), Path("out.mp4"), 0, 8,
        fps=2, max_width=512,
    )
    assert argv[argv.index("-vf") + 1] == (
        "fps=2,scale=512:-2:force_original_aspect_ratio=decrease"
    )


def test_export_paths_are_portable_to_output_manifest_root(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    source_manifest = tmp_path / "source" / "manifest.jsonl"
    source_manifest.parent.mkdir()
    output_root = tmp_path / "different_output"
    target_dir = output_root / "media"

    monkeypatch.setattr(
        "scripts.export_witnessed_reaction_candidate_videos.ffprobe_duration",
        lambda *_args: 10.0,
    )
    monkeypatch.setattr(
        "scripts.export_witnessed_reaction_candidate_videos.subprocess.run",
        lambda argv, check: Path(argv[-1]).write_bytes(b"video"),
    )
    rows = [{
        "candidate_id": "a", "proxy_clip": str(source),
        "media_start_sec": 0, "media_end_sec": 5,
        "candidate_start_sec": 2, "candidate_end_sec": 3,
    }]
    exported = export(
        rows, source_manifest, target_dir, "ffmpeg",
        portable_root=output_root,
    )

    assert exported[0]["candidate_video_path"] == "media/0000.mp4"
