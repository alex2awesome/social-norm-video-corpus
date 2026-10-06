from __future__ import annotations

import json
from pathlib import Path

from src import clip_extract


def test_retain_discussion_video_moves_source_to_stable_storage(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "dailymotion__x1.mp4"
    raw.parent.mkdir()
    raw.write_bytes(b"video-bytes")
    stable = tmp_path / "discussion_video"
    monkeypatch.setattr(clip_extract.state, "resolve_path", lambda value: Path(value))

    result = clip_extract.retain_discussion_video(
        raw,
        "dailymotion__x1",
        {"paths": {"discussion_video": str(stable)}},
    )

    assert result == stable / "dailymotion__x1.mp4"
    assert result.read_bytes() == b"video-bytes"
    assert not raw.exists()


def test_save_discussion_records_retained_video(tmp_path, monkeypatch):
    discussion = tmp_path / "discussion"
    video = tmp_path / "discussion_video" / "dailymotion__x1.mp4"
    monkeypatch.setattr(clip_extract.state, "resolve_path", lambda value: Path(value))

    output = clip_extract.save_discussion(
        "dailymotion__x1",
        [{"matched_text": "That was wrong", "norm": "no theft", "tag": "criticism",
          "start": 1.0, "end": 2.0}],
        {"paths": {"discussion": str(discussion)}},
        {"url": "https://example.test/video", "title": "Example", "source": "test"},
        source_video=video,
    )

    record = json.loads(output.read_text())
    assert record["source_video"] == "dailymotion__x1.mp4"
    assert record["n_statements"] == 1
