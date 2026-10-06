import pytest

from scripts.evaluate_commentary_title_event_expansion import decisions, evaluate


def model_row(index: int, candidate: str = "yes") -> dict:
    return {
        "audit_index": index,
        "error": None,
        "result": {
            "candidate_event": candidate,
            "title_action_visible": "yes" if candidate == "yes" else "uncertain",
            "actor_action_target_same_event": "yes" if candidate == "yes" else "uncertain",
            "candidate_start_sec": 1,
            "candidate_end_sec": 2,
        },
    }


def manual(index: int, event: str, action: str, status: str = "ok") -> dict:
    return {
        "audit_index": str(index),
        "candidate_id": f"c{index}",
        "render_status": status,
        "source_has_candidate_event_footage": event,
        "title_action_visible": action,
    }


def test_decisions_preserve_bounded_uncertain_for_review_only() -> None:
    result = model_row(0, "uncertain")["result"]
    assert decisions(result) == {
        "candidate_yes": False,
        "candidate_or_bounded_uncertain": True,
        "strict_literal": False,
    }


def test_evaluate_excludes_explicit_decode_failure_and_reports_dual_rules() -> None:
    ledger = [
        manual(0, "yes", "yes"),
        manual(1, "no", "no"),
        manual(2, "unknown", "unknown", "decode_failure"),
    ]
    first = [model_row(0), model_row(1)]
    second = [model_row(0), model_row(1, "no")]
    _, report = evaluate(ledger, {"qwen": first, "glm": second})
    assert report["evaluable_rendered_rows"] == 2
    assert report["explicit_render_failures"] == 1
    strict = report["rules"]["strict_literal"]["all_model_consensus"]
    assert strict["title_action_yes"]["precision"] == 1
    assert strict["title_action_yes"]["recall"] == 1


def test_evaluate_fails_on_missing_rendered_model_output() -> None:
    with pytest.raises(ValueError, match="missing qwen output"):
        evaluate([manual(0, "yes", "yes")], {"qwen": []})
