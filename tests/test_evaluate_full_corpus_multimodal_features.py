import numpy as np

from scripts.evaluate_full_corpus_multimodal_features import (
    feature_groups,
    flatten,
    last_successful_rows,
    select_oof_threshold,
)


def test_flatten_keeps_pose_clip_and_xclip_axes():
    row = {
        "low_level": {"motion_mean": 0.2},
        "keypoints": {"person_count_mean": 2},
        "clip_scores": {
            "prompts": ["a", "b"],
            "mean_probabilities": [0.6, 0.4],
            "max_probabilities": [0.8, 0.5],
        },
        "xclip_scores": {
            "prompts": ["c", "d"],
            "probabilities": [0.7, 0.3],
        },
        "transcript_regex": {"reported_speech": 2},
        "transcript_llm": {
            "direct_depiction_prior": 0.8,
            "offscreen_description_prior": 0.1,
            "discourse_mode": "direct_interaction",
        },
    }
    values = flatten(row)
    assert values["low.motion_mean"] == 0.2
    assert values["pose.person_count_mean"] == 2
    assert values["clip.mean_probabilities.0"] == 0.6
    assert values["clip.max_probabilities.1"] == 0.5
    assert values["xclip.probabilities.0"] == 0.7
    assert values["transcript.regex.reported_speech"] == 2
    assert values["transcript.llm.direct_depiction_prior"] == 0.8
    assert values["transcript.discourse.direct_interaction"] == 1
    groups = feature_groups(set(values))
    assert set(groups["all"]) == set(values)
    assert "clip_plus_xclip" in groups
    assert "transcript_all" in groups


def test_feature_groups_do_not_mislabel_clip_as_clip_plus_xclip():
    groups = feature_groups({"clip.mean_probabilities.0"})
    assert "clip" in groups
    assert "clip_plus_xclip" not in groups


def test_last_successful_rows_merges_append_only_sections(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        '{"item_id":"a","low_level":{"motion_mean":0.2},'
        '"keypoints":{"person_count_mean":2},"error":null}\n'
        '{"item_id":"b","low_level":{},"error":"decode"}\n'
    )
    second.write_text(
        '{"item_id":"a","xclip_scores":{"prompts":["x"],'
        '"probabilities":[1.0]},"error":null}\n'
    )
    rows = last_successful_rows([first, second])
    assert rows["a"]["keypoints"]["person_count_mean"] == 2
    assert rows["a"]["xclip_scores"]["probabilities"] == [1.0]
    assert "b" not in rows


def test_oof_threshold_requires_precision_and_minimum_predictions():
    truth = np.asarray([1, 1, 1, 0, 0, 0])
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    selected = select_oof_threshold(
        truth,
        probabilities,
        minimum_precision=0.8,
        minimum_predictions=3,
    )
    assert selected["available"]
    assert selected["selection_metrics"]["threshold"] == 0.7
    assert selected["selection_metrics"]["precision"] == 1.0
