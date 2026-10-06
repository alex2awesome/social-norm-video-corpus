import math

import numpy as np

from scripts.score_witnessed_speaker_embedding_proxy import (
    audio_window,
    cosine_similarity,
    summarize_voice_proxy,
    surrounding_segments,
)


def test_surrounding_segments_selects_nearest_nonoverlapping_context() -> None:
    segments = [
        {"clip_start": 0.0, "clip_end": 1.0, "text": "old"},
        {"clip_start": 7.0, "clip_end": 8.0, "text": "before"},
        {"clip_start": 8.5, "clip_end": 10.5, "text": "overlap"},
        {"clip_start": 11.0, "clip_end": 12.0, "text": "after"},
    ]
    before, after = surrounding_segments(segments, 9.0, 10.0)
    assert [row["text"] for row in before] == ["old", "before"]
    assert [row["text"] for row in after] == ["after"]


def test_cosine_similarity_and_sandwich_proxy() -> None:
    actor = np.asarray([1.0, 0.0])
    bystander = np.asarray([0.0, 1.0])
    assert cosine_similarity(actor, actor) == 1
    result = summarize_voice_proxy(bystander, [actor], [actor])
    assert result["speaker_proxy.new_vs_recent_prior"] == 1
    assert result["speaker_proxy.sandwiched_third_voice"] == 1


def test_voice_proxy_abstains_when_context_missing() -> None:
    result = summarize_voice_proxy(np.asarray([1.0, 0.0]), [], [])
    assert result["speaker_proxy.available"] == 1
    assert math.isnan(result["speaker_proxy.immediate_before_similarity"])
    assert result["speaker_proxy.new_vs_recent_prior"] == 0


def test_audio_window_pads_short_audio() -> None:
    waveform = np.arange(8, dtype=np.float32)
    result = audio_window(waveform, 0.0, 0.5, 8, minimum_seconds=1.0)
    assert result.shape == (8,)
    assert np.array_equal(result[:4], waveform[:4])
    assert np.array_equal(result[4:], np.zeros(4))
