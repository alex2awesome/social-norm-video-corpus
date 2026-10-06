from scripts.evaluate_witnessed_reaction_features import (
    audio_features,
    reaction_segment,
    visual_temporal_features,
)


def test_reaction_segment_prefers_direct_quote():
    segments = [
        {"start": 0, "end": 2, "text": "He was talking loudly."},
        {"start": 2, "end": 4, "text": "Bro please shut your mouth."},
    ]
    segment, score = reaction_segment(
        "Bro, please shut your mouth", "respect bro shut your mouth", segments
    )
    assert segment["start"] == 2
    assert score > 0.9


def test_audio_features_abstains_without_clip(tmp_path):
    result = audio_features(tmp_path / "missing.mp4", 1.0)
    assert result["audio.boundary_available"] == 0.0


def test_visual_temporal_features_abstains_without_clip(tmp_path):
    result = visual_temporal_features(tmp_path / "missing.mp4", 1.0)
    assert result["temporal.boundary_available"] == 0.0
