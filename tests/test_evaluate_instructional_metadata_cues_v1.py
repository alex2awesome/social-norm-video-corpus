import json

import pytest

from scripts.evaluate_instructional_metadata_cues_v1 import cue_values, evaluate, run


def test_cues_are_literal_and_explanation_is_not_excluded() -> None:
    row = {
        "title": "Role-play Scenario: Sharing",
        "found_by_query": "sharing lesson",
        "polarity": "explanation",
        "start_quote": "I asked you to share it",
        "end_quote": "",
        "explanation": "",
    }
    cues = cue_values(row)
    assert cues["title_scene"] is True
    assert cues["dialogue_exchange"] is True
    assert cues["social_action_lexicon"] is True
    assert cues["non_explanation"] is False
    assert cues["scene_and_non_explanation"] is False


def test_evaluate_never_grants_automatic_acceptance() -> None:
    prereg = {
        "policy": "review_only",
        "cues": {"title_scene": "x"},
        "evaluation": {
            "minimum_selected": 1,
            "ranking_gate_visual_precision": 0.5,
            "ranking_gate_visual_precision_delta_over_base": 0.0,
            "ranking_gate_visual_recall": 0.1,
        },
    }
    rows = [
        {
            "uid": "u1",
            "manual_visual_demo": True,
            "manual_exact_original": False,
            "cues": {"title_scene": True},
        },
        {
            "uid": "u2",
            "manual_visual_demo": False,
            "manual_exact_original": False,
            "cues": {"title_scene": False},
        },
    ]
    result = evaluate(rows, prereg)
    assert result["decisions"]["title_scene"]["ranking_gate_passed"] is True
    assert result["decisions"]["title_scene"]["automatic_acceptance"] is False
    assert result["corpus_mutated"] is False


def test_run_fails_closed_on_changed_manual_input(tmp_path) -> None:
    input_path = tmp_path / "manual.jsonl"
    input_path.write_text(json.dumps({"uid": "u"}) + "\n")
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(
        json.dumps(
            {
                "evaluation_cohort": {"records_sha256": "bad", "items": 1},
                "cues": {},
                "evaluation": {},
                "policy": "review_only",
            }
        )
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        run(input_path, prereg_path, tmp_path / "out.jsonl", tmp_path / "summary.json")
