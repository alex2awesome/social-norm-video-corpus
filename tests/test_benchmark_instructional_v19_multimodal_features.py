from scripts.benchmark_instructional_v19_multimodal_features import (
    benchmark,
    feature_groups,
)


def test_feature_groups_are_atomic_and_composed() -> None:
    names = {
        "low.motion_mean",
        "pose.person_count_mean",
        "clip.mean_probabilities.0",
        "xclip.probabilities.0",
        "transcript.regex.reported_speech",
    }
    groups = feature_groups(names)
    assert groups["low_level"] == ["low.motion_mean"]
    assert groups["pose"] == ["pose.person_count_mean"]
    assert groups["clip"] == ["clip.mean_probabilities.0"]
    assert groups["xclip"] == ["xclip.probabilities.0"]
    assert set(groups["low_plus_pose"]) == {
        "low.motion_mean",
        "pose.person_count_mean",
    }
    assert set(groups["clip_plus_xclip"]) == {
        "clip.mean_probabilities.0",
        "xclip.probabilities.0",
    }
    assert "transcript.regex.reported_speech" not in groups["all_visual"]


def test_benchmark_can_report_a_partial_feature_stack() -> None:
    manifest = [
        {
            "item_id": f"i{index}",
            "uid": f"u{index}",
            "audit_cohort": "v18" if index >= 108 else "v15",
            "gold_scene_visible": index % 2 == 0,
            "gold_usable": index % 3 == 0,
            "gold_exact_social_norm": index % 4 == 0,
        }
        for index in range(168)
    ]
    scores = {
        row["item_id"]: {
            "low_level": {
                "motion_mean": float(index % 5),
                "motion_max": float(index % 7),
            }
        }
        for index, row in enumerate(manifest)
    }
    report = benchmark(manifest, scores, 0.90)
    assert report["available_atomic_feature_groups"] == ["low_level"]
    assert set(report["model_results"]) == {"low_level", "all_visual"}
