import pytest

from scripts.evaluate_commentary_title_candidate_audit import evaluate


def test_evaluate_commentary_candidate_cues() -> None:
    selected = [
        {
            "audit_index": index,
            "candidate_id": f"c{index}",
            "title_event_cues": ["actor_action"] if index < 2 else ["capture_before_event"],
        }
        for index in range(3)
    ]
    ledger = [
        {
            "audit_index": str(index),
            "candidate_id": f"c{index}",
            "render_status": "ok",
            "source_has_candidate_event_footage": ("yes", "uncertain", "no")[index],
            "talking_head_only": "no",
            "usable_for_visual_localization_review": ("yes", "uncertain", "no")[index],
            "title_action_visible": ("yes", "no", "no")[index],
            "visual_form": "scene",
            "manual_note": "note",
        }
        for index in range(3)
    ]
    _, summary = evaluate(selected, ledger, expected_count=3)
    assert summary["source_has_candidate_event_footage"]["counts"] == {
        "no": 1,
        "uncertain": 1,
        "yes": 1,
    }
    assert summary["per_title_cue"]["actor_action"]["items"] == 2


def test_evaluate_rejects_candidate_mismatch() -> None:
    with pytest.raises(ValueError, match="candidate mismatch"):
        evaluate(
            [{"audit_index": 0, "candidate_id": "a", "title_event_cues": []}],
            [{"audit_index": "0", "candidate_id": "b"}],
            expected_count=1,
        )
