from scripts.evaluate_witnessed_wwyd_cue_confirmation_v1 import metric, validate, wilson


def test_wilson_perfect_36_clears_preregistered_lower_bound():
    assert wilson(36, 36)[0] >= 0.90


def test_broad_rule_can_fail_confidence_gate_despite_high_point_precision():
    sealed = [{"item_id": str(i)} for i in range(44)]
    post = [
        {
            "source_nonorganic_produced": "yes" if i < 42 else "no",
            "selected_clip_enacted_scenario": "yes",
        }
        for i in range(44)
    ]
    result = metric(sealed, post, lambda row: True)
    assert result["precision"] > 0.95
    assert result["gate_checks"]["minimum_precision"] is True
    assert result["gate_checks"]["minimum_precision_wilson_95_lower"] is False
    assert result["passes_gate"] is False


def test_validation_rejects_demo_without_enacted_scene():
    sealed = [{"audit_index": 0, "channel": "What Would You Do?"}]
    blind = [{
        "audit_index": "0",
        "deliberately_produced_setup": "yes",
        "contains_enacted_social_scenario": "yes",
        "instructional_demo_candidate": "yes",
        "visual_evidence": "actor card",
    }]
    post = [{
        "audit_index": "0",
        "official_wwyd_channel": "yes",
        "source_nonorganic_produced": "yes",
        "selected_clip_enacted_scenario": "no",
        "instructional_demo_review": "yes",
        "recommended_route": "instructional_demo_review",
        "post_reveal_evidence": "official source",
    }]
    try:
        validate(sealed, blind, post)
    except ValueError as error:
        assert "demo review without enacted scene" in str(error)
    else:
        raise AssertionError("validation should fail")
