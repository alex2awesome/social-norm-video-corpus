import pytest
import numpy as np

from scripts.materialize_witnessed_corpus_audit_media import (
    ffmpeg_argv,
    flatten_selection,
    render_storyboard,
)


def selection():
    return [{
        "clip_audit_index": 4,
        "item_id": "witnessed:youtube__a:0",
        "uid": "youtube__a",
        "cohort": "predicted_positive_enrichment",
        "model_scores": {"secret": "must not leak"},
        "candidates": [{
            "candidate_id": "candidate-a",
            "source_clip": "/source/a.mp4",
            "media_start_sec": 2,
            "media_end_sec": 8,
            "candidate_relative_start_sec": 2,
            "candidate_relative_end_sec": 3,
            "candidate_text": "hidden ASR",
        }],
    }]


def test_flatten_selection_omits_model_cohort_and_asr_fields():
    rows = flatten_selection(selection())
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "candidate-a"
    assert "model_scores" not in rows[0]
    assert "cohort" not in rows[0]
    assert "candidate_text" not in rows[0]


def test_ffmpeg_retains_audio_and_uses_original_interval(tmp_path):
    row = flatten_selection(selection())[0]
    argv = ffmpeg_argv("/bin/ffmpeg", row, tmp_path / "out.mp4")
    assert argv[argv.index("-ss") + 1] == "2.000"
    assert argv[argv.index("-t") + 1] == "6.000"
    assert "-an" not in argv
    assert argv[argv.index("-c:a") + 1] == "aac"
    assert "fps=12" in argv[argv.index("-vf") + 1]


def test_flatten_selection_rejects_duplicate_candidate_ids():
    rows = selection()
    rows.append({**rows[0], "clip_audit_index": 5})
    with pytest.raises(ValueError, match="duplicate"):
        flatten_selection(rows)


def test_render_storyboard_records_dense_blind_lineage(tmp_path, monkeypatch):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(24)]
    timestamps = [index / 2 for index in range(24)]
    monkeypatch.setattr(
        "scripts.materialize_witnessed_corpus_audit_media.sample_frames",
        lambda _path, count: (
            frames,
            {"sampled_timestamps": timestamps, "sampled_frames": count},
        ),
    )
    observed = {}

    def fake_sheet(given_frames, given_timestamps, audit_index):
        observed.update({
            "frames": len(given_frames),
            "timestamps": given_timestamps,
            "audit_index": audit_index,
        })
        return np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "scripts.materialize_witnessed_corpus_audit_media.make_dense_sheet",
        fake_sheet,
    )
    target = tmp_path / "storyboards" / "0007.jpg"
    result = render_storyboard(
        tmp_path / "blind.mp4",
        target,
        audit_candidate_index=7,
        requested_frames=24,
    )
    assert observed == {
        "frames": 24,
        "timestamps": timestamps,
        "audit_index": 7,
    }
    assert result["storyboard_path"] == str(target.resolve())
    assert result["storyboard_rendered_frames"] == 24
    assert result["storyboard_sampled_timestamps_sec"] == timestamps
    assert len(result["storyboard_sha256"]) == 64
    assert not any(
        token in result for token in ("model_scores", "cohort", "candidate_text")
    )


def test_render_storyboard_rejects_missing_or_unaligned_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.materialize_witnessed_corpus_audit_media.sample_frames",
        lambda _path, _count: ([np.zeros((2, 2, 3))], {"sampled_timestamps": []}),
    )
    with pytest.raises(ValueError, match="no aligned frames"):
        render_storyboard(
            tmp_path / "blind.mp4",
            tmp_path / "blind.jpg",
            audit_candidate_index=0,
            requested_frames=24,
        )


@pytest.mark.parametrize("count", [0, 11, 13, 25])
def test_render_storyboard_rejects_invalid_frame_count(tmp_path, count):
    with pytest.raises(ValueError, match="positive multiple of 12"):
        render_storyboard(
            tmp_path / "blind.mp4",
            tmp_path / "blind.jpg",
            audit_candidate_index=0,
            requested_frames=count,
        )
