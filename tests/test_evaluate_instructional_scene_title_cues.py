from scripts.evaluate_instructional_scene_title_cues import evaluate, title_cues


def test_title_cues_select_scene_language_without_presenter_collision() -> None:
    result = title_cues("Animated Social Story: Taking Turns")
    assert result["scene_title_candidate"] is True
    assert "animation_story" in result["positive_cues"]


def test_title_cues_fail_closed_on_presenter_language() -> None:
    result = title_cues("Animated Story Explained: How to Teach Sharing")
    assert result["positive_cues"]
    assert result["negative_presenter_cue"] is True
    assert result["scene_title_candidate"] is False


def test_evaluate_keeps_visual_and_exact_targets_separate() -> None:
    rows = [
        {
            "cohort": "a",
            "uid": "u1",
            "scene_title_candidate": True,
            "manual_visual_demo": True,
            "manual_exact_demo": False,
            "positive_cues": ["roleplay_scenario"],
            "negative_presenter_cue": False,
        },
        {
            "cohort": "a",
            "uid": "u2",
            "scene_title_candidate": False,
            "manual_visual_demo": False,
            "manual_exact_demo": False,
            "positive_cues": [],
            "negative_presenter_cue": False,
        },
    ]
    summary = evaluate(rows)
    assert summary["aggregate"]["visual_demo"]["precision"] == 1.0
    assert summary["aggregate"]["exact_demo"]["precision"] == 0.0
    assert summary["policy"].endswith("never_a_label")
