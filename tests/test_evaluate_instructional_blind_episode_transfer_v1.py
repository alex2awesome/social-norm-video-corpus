from scripts.evaluate_instructional_blind_episode_transfer_v1 import assess_gate


def fixtures(precision=0.85, recall=0.5, selected=20):
    evaluation = {
        "items": 100,
        "manual_coverage_complete": True,
        "manual_visual_demos": 25,
        "manual_usable_demos": 20,
        "episode_without_completeness_development": {
            "qwen": {"precision": precision, "recall": recall, "selected": selected}
        },
    }
    preregistration = {
        "fresh_validation_gate": {
            "minimum_clips": 100,
            "minimum_precision": 0.8,
            "minimum_recall": 0.4,
            "minimum_selected": 15,
            "required_manual_coverage": 100,
            "target": "connected_visual_demonstration_in_any_valid_medium",
        },
        "frozen_rule": {"model": "qwen", "prompt": "blind_episode_v1"},
    }
    audits = [
        {
            "episode_output_review": "ok",
            "symbolic_output_review": "ok",
        }
        for _ in range(100)
    ]
    return evaluation, preregistration, audits


def test_gate_pass_is_shadow_routing_only() -> None:
    result = assess_gate(*fixtures())
    assert result["preregistered_pass"] is True
    assert result["status"] == "passed_for_shadow_candidate_routing_only"
    assert result["automatic_acceptance"] is False
    assert result["corpus_mutation_authorized"] is False


def test_gate_fails_when_output_audit_is_incomplete() -> None:
    evaluation, preregistration, audits = fixtures()
    result = assess_gate(evaluation, preregistration, audits[:-1])
    assert result["checks"]["manual_model_output_audit"] is False
    assert result["preregistered_pass"] is False


def test_gate_fails_frozen_precision_threshold() -> None:
    result = assess_gate(*fixtures(precision=0.79))
    assert result["checks"]["minimum_precision"] is False
    assert result["preregistered_pass"] is False
