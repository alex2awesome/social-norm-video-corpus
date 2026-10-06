from scripts.evaluate_witnessed_video_benchmark import evaluate, predict


def result(*, witnessed="yes", proposed="yes", depiction="organic_scene"):
    return {
        "result": {
            "social_norm_domain": "yes",
            "situated_social_scenario_visible": "yes",
            "observable_social_behavior_or_speech": "yes",
            "usable_demo_after_relabel": "yes",
            "proposed_norm_supported": proposed,
            "procedural_or_nonsocial_activity_only": "no",
            "presentation_or_context_only": "no",
            "witnessed_action_then_reaction_organic": witnessed,
            "localization_quality": "clean",
            "depiction_type": depiction,
        }
    }


def test_routes_keep_current_recovery_and_instructional_separate():
    recovered = result(proposed="no")
    assert not predict(recovered, "strict_current", allow_broad=False)
    assert predict(recovered, "witnessed_recovery", allow_broad=False)
    staged = result(witnessed="no", proposed="yes", depiction="enacted_scene")
    assert predict(staged, "instructional_reroute", allow_broad=False)
    assert predict(staged, "any_visual_recovery", allow_broad=False)


def test_intersection_requires_both_models():
    benchmark = [
        {
            "item_id": "a",
            "gold_strict_current": False,
            "gold_witnessed_recovery": True,
            "gold_instructional_reroute": False,
            "gold_any_visual_recovery": True,
        },
        {
            "item_id": "b",
            "gold_strict_current": False,
            "gold_witnessed_recovery": False,
            "gold_instructional_reroute": False,
            "gold_any_visual_recovery": False,
        },
    ]
    primary = [
        {"item_id": "a", **result()},
        {"item_id": "b", **result()},
    ]
    secondary = [
        {"item_id": "a", **result()},
        {
            "item_id": "b",
            **result(witnessed="no", depiction="organic_scene"),
        },
    ]
    report = evaluate(benchmark, primary, secondary)
    metrics = report["rules"]["witnessed_recovery_clean"]["intersection"]
    assert metrics["predicted_positive_items"] == ["a"]
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_errors_abstain_instead_of_becoming_negative():
    row = {"error": "transport failure", "result": None}
    assert predict(row, "witnessed_recovery", allow_broad=False) is None
