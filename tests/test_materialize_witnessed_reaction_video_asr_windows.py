from pathlib import Path

import pytest

from scripts.materialize_witnessed_reaction_video_asr_windows import (
    canonical_item_id,
    cut_proxy_ffmpeg_argv,
    ffmpeg_argv,
    flatten_candidates,
    full_proxy_ffmpeg_argv,
    sampling_fps,
    source_from_candidate_parent,
)


def test_canonical_item_id_joins_candidate_and_source_manifests():
    assert canonical_item_id("witnessed:uid:clip_12") == "witnessed:uid:12"
    assert canonical_item_id("witnessed:uid:12") == "witnessed:uid:12"


def test_sampling_fps_caps_full_clip_at_96_frames():
    assert sampling_fps(24, 3, 96) == 3
    assert sampling_fps(96, 3, 96) == 1
    with pytest.raises(ValueError):
        sampling_fps(0, 3, 96)


def test_flatten_excludes_negative_candidates_and_preserves_context(monkeypatch):
    monkeypatch.setattr(
        "scripts.materialize_witnessed_reaction_video_asr_windows.load_clip_segments",
        lambda _row: [
            {"text": "act"}, {"text": "stop"}, {"text": "after"}
        ],
    )
    parents = [{
        "item_id": "witnessed:u:clip_0", "uid": "u",
        "candidates": [
            {"segment_index": 1, "text": "stop", "start": 2, "end": 3,
             "window_start": 0, "window_end": 6, "mechanisms": ["direct"],
             "negative_self_defense": False, "negative_reported": False},
            {"segment_index": 2, "text": "I did nothing", "start": 3, "end": 4,
             "window_start": 0, "window_end": 7, "mechanisms": ["direct"],
             "negative_self_defense": True, "negative_reported": False},
        ],
    }]
    sources = [{"item_id": "witnessed:u:0", "source_clip": "/x.mp4"}]
    rows, missing = flatten_candidates(parents, sources)
    assert not missing
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "witnessed:u:clip_0:candidate_1"
    assert rows[0]["context_before"] == ["act"]
    assert rows[0]["context_after"] == ["after"]


def test_explicit_frozen_context_overrides_missing_transcript_context(monkeypatch):
    monkeypatch.setattr(
        "scripts.materialize_witnessed_reaction_video_asr_windows.load_clip_segments",
        lambda _row: [],
    )
    parents = [{
        "item_id": "witnessed:u:0", "uid": "u",
        "candidates": [{
            "segment_index": 1, "text": "stop", "start": 2, "end": 3,
            "window_start": 0, "window_end": 6, "mechanisms": ["direct"],
            "negative_self_defense": False, "negative_reported": False,
        }],
    }]
    sources = [{"item_id": "witnessed:u:0", "source_clip": "/x.mp4"}]
    contexts = {
        "witnessed:u:0:candidate_1": {
            "context_before": ["earlier action"],
            "context_after": ["later response"],
        }
    }
    rows, _missing = flatten_candidates(parents, sources, contexts)
    assert rows[0]["context_before"] == ["earlier action"]
    assert rows[0]["context_after"] == ["later response"]


def test_missing_inventory_source_recovers_from_candidate_metadata(tmp_path):
    hit = tmp_path / "data" / "hits" / "u"
    hit.mkdir(parents=True)
    metadata = hit / "metadata.json"
    metadata.write_text("{}")
    clip = hit / "clip_2.mp4"
    clip.write_bytes(b"video")
    parent = {
        "item_id": "witnessed:u:clip_2",
        "metadata_path": str(metadata),
        "clip_name": "clip_2.mp4",
    }
    recovered = source_from_candidate_parent(parent)
    assert recovered is not None
    assert recovered["item_id"] == "witnessed:u:2"
    assert recovered["source_clip"] == str(clip.resolve())


def test_ffmpeg_contract_is_silent_frame_capped_proxy_representation():
    argv = ffmpeg_argv(
        "ffmpeg", Path("in.mp4"), Path("out.mp4"), source_start=0,
        window_start=5, window_duration=8, fps=1.5, max_width=512,
    )
    assert "-an" in argv
    filters = argv[argv.index("-vf") + 1]
    assert filters.startswith("fps=1.500000,scale=512")
    assert "scale=trunc(iw/2)*2:trunc(ih/2)*2" in filters


def test_two_stage_contract_caps_full_proxy_then_cuts_without_resampling():
    full = full_proxy_ffmpeg_argv(
        "ffmpeg", Path("source.mp4"), Path("proxy.mp4"), source_start=0,
        duration=80, fps=1.2, max_frames=96, max_width=512,
    )
    assert full[full.index("-frames:v") + 1] == "96"
    assert "fps=1.200000" in full[full.index("-vf") + 1]
    cut = cut_proxy_ffmpeg_argv(
        "ffmpeg", Path("proxy.mp4"), Path("candidate.mp4"),
        window_start=5, window_duration=8,
    )
    assert "-vf" not in cut
    assert cut[cut.index("-ss") + 1] == "5.000"
