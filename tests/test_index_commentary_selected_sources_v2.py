from pathlib import Path

import pytest

import scripts.index_commentary_selected_sources_v2 as module


def test_indexes_one_retained_source(monkeypatch, tmp_path: Path):
    video = tmp_path / "youtube__abc.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(module, "probe_duration", lambda *_: 12.5)
    rows, failures = module.index(
        [{"uid": "youtube__abc"}], tmp_path, "ffprobe", 1800
    )
    assert failures == []
    assert rows[0]["duration_sec"] == 12.5
    assert rows[0]["source_path"] == str(video.resolve())


def test_records_overlong_source_as_explicit_failure(monkeypatch, tmp_path: Path):
    (tmp_path / "youtube__abc.mp4").write_bytes(b"video")
    monkeypatch.setattr(module, "probe_duration", lambda *_: 1801)
    rows, failures = module.index(
        [{"uid": "youtube__abc"}], tmp_path, "ffprobe", 1800
    )
    assert rows == []
    assert "exceeds audit limit" in failures[0]["error"]


def test_rejects_duplicate_uids(tmp_path: Path):
    with pytest.raises(ValueError, match="duplicate"):
        module.index([{"uid": "u"}, {"uid": "u"}], tmp_path, "ffprobe", 1800)
