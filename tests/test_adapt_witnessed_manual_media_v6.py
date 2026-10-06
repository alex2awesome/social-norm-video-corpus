import pytest

from scripts.adapt_witnessed_manual_media_v6 import adapt


def selection():
    return [{
        "item_id": "witnessed:u:clip_0",
        "uid": "u",
        "cohort": "v3_positive_enrichment",
        "candidates": [{
            "candidate_id": "witnessed:u:clip_0:candidate_1",
            "candidate_text": "Stop doing that",
            "candidate_relative_start_sec": 1.0,
            "candidate_relative_end_sec": 2.0,
            "window_duration_sec": 4.0,
        }],
    }]


def media():
    return [{
        "candidate_id": "witnessed:u:clip_0:candidate_1",
        "manual_media_path": "/tmp/a.mp4",
        "manual_media_sha256": "abc",
        "audio_present": True,
        "error": None,
    }]


def test_restores_only_candidate_context_and_new_media():
    rows = adapt(selection(), media())
    assert rows[0]["candidate_text"] == "Stop doing that"
    assert rows[0]["candidate_video_path"] == "/tmp/a.mp4"
    assert rows[0]["audio_present"] is True
    assert "cohort" not in rows[0]
    assert rows[0]["prior_model_prediction_exposed"] is False


def test_requires_exact_media_coverage():
    with pytest.raises(ValueError, match="exactly cover"):
        adapt(selection(), [])


def test_rejects_failed_media():
    failed = media()
    failed[0]["error"] = "decode failed"
    with pytest.raises(ValueError, match="manual media failed"):
        adapt(selection(), failed)
