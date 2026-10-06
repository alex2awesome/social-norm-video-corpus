from scripts.merge_shadow_scene_features import merge_rows


def test_merge_rows_preserves_complementary_non_null_sections():
    base = [
        {
            "item_id": "a",
            "low_level": {"motion": 1},
            "clip_scores": None,
            "xclip_scores": None,
        }
    ]
    clip = [
        {
            "item_id": "a",
            "low_level": {"motion": 2},
            "clip_scores": {"probabilities": [0.8]},
            "xclip_scores": None,
        }
    ]
    xclip = [
        {
            "item_id": "a",
            "clip_scores": None,
            "xclip_scores": {"probabilities": [0.7]},
        }
    ]

    assert merge_rows(base, [clip, xclip]) == [
        {
            "item_id": "a",
            "low_level": {"motion": 2},
            "clip_scores": {"probabilities": [0.8]},
            "xclip_scores": {"probabilities": [0.7]},
        }
    ]
