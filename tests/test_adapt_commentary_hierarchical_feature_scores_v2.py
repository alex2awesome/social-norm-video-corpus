import pytest

from scripts.adapt_commentary_hierarchical_feature_scores_v2 import (
    CLIP_SOCIAL_PROMPT,
    adapt,
    adapt_one,
)


def baseline(item_id="w"):
    return {
        "item_id": item_id,
        "error": None,
        "low_level": {
            "motion_mean": 0.2,
            "histogram_delta_mean": 0.3,
            "hard_cut_fraction": 0.1,
        },
        "keypoints": {
            "multiple_people_fraction": 0.4,
            "close_pair_fraction": 0.6,
            "wrist_near_other_person_fraction": 0.5,
        },
        "clip_scores": {
            "prompts": [CLIP_SOCIAL_PROMPT, "talking head"],
            "mean_probabilities": [0.7, 0.3],
        },
        "xclip_scores": None,
    }


def test_adapter_maps_existing_baselines_without_thresholding():
    row = adapt_one(baseline())
    assert row == {
        "window_id": "w",
        "motion": 0.2,
        "scene_change": 0.3,
        "person_interaction": 0.6,
        "social_scene_similarity": 0.7,
        "error": None,
        "policy": "candidate_ranking_only_no_keep_reject_or_corpus_mutation",
    }


def test_xclip_and_clip_are_complementary_max_rankers():
    row = baseline()
    row["xclip_scores"] = {
        "prompts": [
            "a situated social interaction between two or more people",
            "text graphic",
        ],
        "probabilities": [0.9, 0.1],
    }
    assert adapt_one(row)["social_scene_similarity"] == 0.9


def test_missing_required_section_is_explicit_error_not_negative_score():
    row = baseline()
    row["keypoints"] = None
    output = adapt_one(row)
    assert output["error"].startswith("ValueError:")
    assert "person_interaction" not in output


def test_upstream_decode_error_is_preserved():
    row = baseline()
    row["error"] = "decode failed"
    assert adapt_one(row) == {"window_id": "w", "error": "decode failed"}


def test_duplicate_window_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        adapt([baseline(), baseline()])
