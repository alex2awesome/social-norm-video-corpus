import pytest

from scripts.evaluate_commentary_dual_retrieval_transfer_v1 import evaluate


MODELS = ["qwen3-vl-8b-instruct", "gemma-3-27b-it"]


def contract(minimum_selected=1):
    return {
        "models": MODELS,
        "gate": {
            "minimum_evaluable_rendered_items": 1, "minimum_selected": minimum_selected,
            "minimum_precision": 0.8, "minimum_precision_wilson_95_lower": 0.0,
        },
    }


def ledger(index=0, usable="yes"):
    return [{
        "audit_index": str(index), "render_status": "ok",
        "usable_for_visual_localization_review": usable,
    }]


def model(index=0, passes=True):
    value = "yes" if passes else "no"
    return [{
        "audit_index": index, "error": None,
        "result": {
            "candidate_event": value, "title_action_visible": value,
            "title_actor_visible": value, "title_target_or_property_visible": value,
            "actor_action_target_same_event": value,
        },
    }]


def audits(index=0):
    return [{
        "audit_index": str(index), "model": name,
        "output_visually_supported": "yes", "error_mechanism": "none",
        "manual_evidence": "The model statement matches the visible storyboard.",
    } for name in MODELS]


def test_exact_dual_conjunction_can_retain_review_ranking_only():
    report = evaluate(
        ledger(), {name: model() for name in MODELS}, audits(), contract()
    )
    assert report["metrics"]["precision"] == 1.0
    assert report["review_ranking_rule_retained"] is True
    assert report["allowed_use"] == "review_ranking_only"
    assert report["automatic_acceptance"] is False


def test_one_model_failure_breaks_conjunction_and_gate():
    outputs = {MODELS[0]: model(), MODELS[1]: model(passes=False)}
    report = evaluate(ledger(), outputs, audits(), contract())
    assert report["metrics"]["selected"] == 0
    assert report["review_ranking_rule_retained"] is False


def test_manual_output_audit_must_exactly_cover_both_models():
    with pytest.raises(ValueError, match="does not exactly cover"):
        evaluate(
            ledger(), {name: model() for name in MODELS}, audits()[:1], contract()
        )


def test_model_set_is_frozen():
    with pytest.raises(ValueError, match="model set"):
        evaluate(ledger(), {MODELS[0]: model()}, audits(), contract())
