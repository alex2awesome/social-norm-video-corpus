from argparse import Namespace

from scripts.score_transcript_scene_priors import (
    aligned_transcript_text,
    regex_features,
    score_one,
)


def test_aligned_transcript_text_joins_nonempty_segments():
    row = {
        "aligned_transcript": [
            {"start": 0, "end": 1, "text": " Please stop. "},
            {"start": 1, "end": 2, "text": ""},
            {"start": 2, "end": 3, "text": "I'm sorry."},
        ]
    }
    assert aligned_transcript_text(row) == "Please stop. I'm sorry."


def test_score_one_prefers_frozen_aligned_transcript(tmp_path):
    row = {
        "item_id": "instructional:a:0",
        "uid": "a",
        "pillar": "instructional",
        "gold_scene_visible": True,
        "gold_social_scene_visible": True,
        "gold_label_matched_visible": False,
        "gold_usable": False,
        "aligned_transcript": [{"text": "Let's watch this scenario."}],
    }
    args = Namespace(
        transcripts=tmp_path,
        padding=5,
        endpoint=None,
        model="unused",
        timeout=1,
    )
    result = score_one(row, args)
    assert result["error"] is None
    assert result["transcript_source"] == "manifest_aligned"
    assert result["regex"]["watch_transition"] == 1


def test_regex_features_separate_demo_and_offscreen_language():
    values = regex_features(
        "According to police, it happened last year. "
        "Now let's watch this role-play. Please stop it."
    )
    assert values["attribution"] == 1
    assert values["retrospective"] == 2
    assert values["explicit_demo"] == 1
    assert values["watch_transition"] == 1
    assert values["direct_exchange"] >= 1
