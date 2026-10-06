import pytest

from scripts.evaluate_instructional_metadata_cues_replication_v1 import evaluate


def _row(uid: str, cue: bool, visual: bool) -> dict:
    return {
        "uid": uid,
        "manual_visual_demo": visual,
        "manual_exact_original": visual,
        "cues": {"cue": cue},
    }


def _configs() -> tuple[dict, dict]:
    parent = {
        "policy": "review_only",
        "cues": {"cue": "test"},
        "evaluation": {
            "minimum_selected": 1,
            "ranking_gate_visual_precision": 0.5,
            "ranking_gate_visual_precision_delta_over_base": 0.0,
            "ranking_gate_visual_recall": 0.1,
        },
    }
    amendment = {"replication_gate": {"minimum_total_selected": 2}}
    return parent, amendment


def test_one_cohort_win_cannot_promote() -> None:
    parent, amendment = _configs()
    development = [_row("d1", True, False), _row("d2", False, True)]
    evaluation = [_row("e1", True, True), _row("e2", False, False)]
    result = evaluate(development, evaluation, parent, amendment)
    decision = result["decisions"]["cue"]
    assert decision["cohort_gate_passed"] == {
        "development_100": False,
        "evaluation_60": True,
    }
    assert decision["replicated_ranking_gate_passed"] is False
    assert decision["automatic_acceptance"] is False


def test_source_overlap_fails_closed() -> None:
    parent, amendment = _configs()
    with pytest.raises(ValueError, match="source overlap"):
        evaluate([_row("same", True, True)], [_row("same", True, True)], parent, amendment)
