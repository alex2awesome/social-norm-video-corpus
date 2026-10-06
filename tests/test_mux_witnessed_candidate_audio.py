from pathlib import Path

from scripts.mux_witnessed_candidate_audio import (
    attach_source_context,
    candidate_ordinal,
    mux_argv,
    video_only_argv,
)


def test_attach_source_context_joins_by_candidate_not_row_order():
    rows = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    contexts = [
        {"candidate_id": "b", "source_clip": "b.mp4"},
        {"candidate_id": "a", "source_clip": "a.mp4"},
    ]
    joined = attach_source_context(rows, contexts)
    assert [row["source_clip"] for row in joined] == ["a.mp4", "b.mp4"]


def test_candidate_ordinal_accepts_new_and_legacy_manifest_schemas():
    assert candidate_ordinal({"ordinal": 3}) == 3
    assert candidate_ordinal({"storyboard_index": 4}) == 4


def test_mux_argv_copies_video_and_cuts_original_audio():
    argv = mux_argv(
        "ffmpeg",
        Path("silent.mp4"),
        Path("source.mp4"),
        Path("av.mp4"),
        audio_start=12.5,
        duration=4.25,
    )
    assert argv[argv.index("-c:v") + 1] == "copy"
    assert argv[argv.index("-c:a") + 1] == "aac"
    assert argv[argv.index("-ss") + 1] == "12.500"
    assert argv[argv.index("-t") + 1] == "4.250"
    assert "1:a:0" in argv


def test_video_only_fallback_does_not_invent_audio():
    argv = video_only_argv("ffmpeg", Path("silent.mp4"), Path("copy.mp4"))
    assert "-c:a" not in argv
    assert argv[argv.index("-c:v") + 1] == "copy"
