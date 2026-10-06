from scripts.evaluate_commentary_scene_benchmark import (
    evaluate,
    evaluate_ensemble,
    manual_positive,
    merge_model_rows,
    predict,
)


def test_manual_positive_requires_target_route_and_optionally_clear_demo():
    row = {
        "expected_disposition": "dense_followup_commentary",
        "demo_quality": "incomplete_or_ambiguous",
    }
    assert manual_positive(row, strict_clear=False)
    assert not manual_positive(row, strict_clear=True)
    assert not manual_positive(
        {
            "expected_disposition": "dense_source_mining_review",
            "demo_quality": "clear_visual",
        },
        strict_clear=False,
    )


def test_rubric_predictions_and_errors():
    assert predict(
        {
            "result": {
                "visually_observable_event": "yes",
                "presentation_or_context_only": "no",
            }
        },
        "v6_blind",
    )
    assert not predict(
        {
            "result": {
                "visually_observable_event": "yes",
                "presentation_or_context_only": "yes",
            }
        },
        "v6_blind",
    )
    assert predict(
        {
            "result": {
                "situated_social_scenario_visible": "yes",
                "concrete_action_or_situated_speech_visible": "yes",
                "proposed_norm_plausibly_demonstrated": "yes",
            }
        },
        "v3_conditioned",
    )
    assert predict({"error": "transport failure"}, "v3_conditioned") is None


def test_retry_rows_replace_base_rows_by_item_id():
    assert merge_model_rows(
        [{"item_id": "a", "error": "too long"}, {"item_id": "b", "result": {}}],
        [{"item_id": "a", "result": {"ok": True}}],
    ) == [
        {"item_id": "a", "result": {"ok": True}},
        {"item_id": "b", "result": {}},
    ]


def test_evaluation_tracks_false_positives_false_negatives_and_abstentions():
    manual = [
        {
            "item_id": "tp",
            "expected_disposition": "dense_followup_commentary",
            "demo_quality": "clear_visual",
        },
        {
            "item_id": "fp",
            "expected_disposition": "text_only",
            "demo_quality": "none",
        },
        {
            "item_id": "fn",
            "expected_disposition": "dense_followup_instructional",
            "demo_quality": "clear_audiovisual",
        },
        {
            "item_id": "err",
            "expected_disposition": "text_only",
            "demo_quality": "none",
        },
    ]
    model = [
        {
            "item_id": "tp",
            "result": {
                "situated_social_scenario_visible": "yes",
                "concrete_action_or_situated_speech_visible": "yes",
                "proposed_norm_plausibly_demonstrated": "yes",
            },
        },
        {
            "item_id": "fp",
            "result": {
                "situated_social_scenario_visible": "yes",
                "concrete_action_or_situated_speech_visible": "yes",
                "proposed_norm_plausibly_demonstrated": "yes",
            },
        },
        {
            "item_id": "fn",
            "result": {
                "situated_social_scenario_visible": "yes",
                "concrete_action_or_situated_speech_visible": "yes",
                "proposed_norm_plausibly_demonstrated": "no",
            },
        },
        {"item_id": "err", "error": "too long"},
    ]

    report = evaluate(manual, model, rubric="v3_conditioned")

    assert report["metrics"]["strict_clear_targets"] == {
        "gold_positive": 2,
        "evaluated": 3,
        "abstained_or_error": 1,
        "predicted_positive": 2,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 0,
        "precision": 0.5,
        "recall": 0.5,
        "predicted_positive_items": ["tp", "fp"],
        "true_positive_items": ["tp"],
        "false_positive_items": ["fp"],
        "false_negative_items": ["fn"],
        "true_negative_items": [],
    }


def test_intersection_ensemble_requires_both_models_to_accept():
    manual = [
        {
            "item_id": "both",
            "expected_disposition": "dense_followup_commentary",
            "demo_quality": "clear_visual",
        },
        {
            "item_id": "one",
            "expected_disposition": "text_only",
            "demo_quality": "none",
        },
    ]

    def row(item_id, value):
        return {
            "item_id": item_id,
            "result": {
                "situated_social_scenario_visible": "yes" if value else "no",
                "concrete_action_or_situated_speech_visible": "yes" if value else "no",
                "proposed_norm_plausibly_demonstrated": "yes" if value else "no",
            },
        }

    report = evaluate_ensemble(
        manual,
        [row("both", True), row("one", True)],
        [row("both", True), row("one", False)],
        rubric="v3_conditioned",
        operator="intersection",
    )

    assert report["metrics"]["strict_clear_targets"]["predicted_positive"] == 1
    assert report["metrics"]["strict_clear_targets"]["true_positive"] == 1
    assert report["metrics"]["strict_clear_targets"]["false_positive"] == 0


def test_visual_recovery_includes_recut_and_source_mining_routes():
    manual = [
        {
            "item_id": "direct",
            "expected_disposition": "dense_followup_commentary",
            "demo_quality": "clear_visual",
        },
        {
            "item_id": "recut",
            "expected_disposition": "dense_recut_review",
            "demo_quality": "incomplete_or_ambiguous",
        },
        {
            "item_id": "mine",
            "expected_disposition": "dense_source_mining_review",
            "demo_quality": "incomplete_or_ambiguous",
        },
        {
            "item_id": "reject",
            "expected_disposition": "reject_visual_target",
            "demo_quality": "none",
        },
    ]
    row = lambda item_id: {
        "item_id": item_id,
        "result": {
            "situated_social_scenario_visible": "yes",
            "concrete_action_or_situated_speech_visible": "yes",
            "proposed_norm_plausibly_demonstrated": "yes",
        },
    }

    report = evaluate(
        manual,
        [row("direct"), row("recut"), row("mine"), row("reject")],
        rubric="v3_conditioned",
    )

    recovery = report["metrics"]["visual_recovery_candidates"]
    assert recovery["gold_positive"] == 3
    assert recovery["true_positive_items"] == ["direct", "recut", "mine"]
    assert recovery["false_positive_items"] == ["reject"]
