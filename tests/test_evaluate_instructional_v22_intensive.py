from scripts.evaluate_instructional_v22_intensive import evaluate, metric, wilson


def output(item_id: str, key: str, decision: str) -> dict:
    return {"item_id": item_id, "error": None, "result": {key: decision}}


def test_metric_has_wilson_intervals() -> None:
    result = metric([True, True, False], [True, False, True])
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["precision_wilson_95"] == wilson(1, 2)


def test_two_stage_evaluation_and_slices() -> None:
    blind = [
        {"candidate_id": f"c{i}", "item_id": f"i{i}", "uid": f"dailymotion__u{i}"}
        for i in range(3)
    ]
    semantic = [
        {
            **row,
            "source_platform": "dailymotion",
            "category": "classroom" if i < 2 else "family",
            "polarity": "correct",
        }
        for i, row in enumerate(blind)
    ]
    ledger = [
        {
            "candidate_id": f"c{i}",
            "item_id": f"i{i}",
            "visual_form": "animation" if i == 0 else "live_action",
            "visual_demo": "yes" if i < 2 else "no",
            "semantic_alignment": "yes" if i == 0 else "no",
            "complete_demo": "yes" if i == 0 else "no",
            "usable_demo": "yes" if i == 0 else "no",
            "note": "manual",
        }
        for i in range(3)
    ]
    visual = {
        "qwen": [output("i0", "demo_pass", "yes"), output("i1", "demo_pass", "yes"), output("i2", "demo_pass", "no")],
        "glm": [output("i0", "demo_pass", "yes"), output("i1", "demo_pass", "no"), output("i2", "demo_pass", "no")],
    }
    alignment = {
        "qwen": [output("i0", "semantic_pass", "yes"), output("i1", "semantic_pass", "no"), output("i2", "semantic_pass", "no")],
        "glm": [output("i0", "semantic_pass", "yes"), output("i1", "semantic_pass", "no"), output("i2", "semantic_pass", "no")],
    }
    rows, summary = evaluate(blind, semantic, ledger, visual, alignment, 3)
    assert rows[0]["pipeline_consensus"] is True
    assert summary["pipeline_consensus"]["precision"] == 1.0
    assert summary["pipeline_consensus"]["recall"] == 1.0
    assert summary["source_cluster_exact"]["precision"] == 1.0
    assert "animation" in summary["slices"]["manual_visual_form"]


def test_evaluation_requires_complete_manual_coverage() -> None:
    try:
        evaluate([], [], [], {"q": []}, {"q": []}, 1)
    except ValueError as exc:
        assert "expected 1 rows" in str(exc)
    else:
        raise AssertionError("expected coverage validation failure")


def test_evaluation_can_fail_closed_on_model_parse_error() -> None:
    blind = [
        {"candidate_id": f"c{i}", "item_id": f"i{i}", "uid": f"dailymotion__u{i}"}
        for i in range(2)
    ]
    semantic = [
        {**row, "source_platform": "dailymotion", "category": "family", "polarity": "correct"}
        for row in blind
    ]
    ledger = [
        {
            "candidate_id": f"c{i}",
            "item_id": f"i{i}",
            "visual_form": "live_action",
            "visual_demo": "yes" if i == 0 else "no",
            "semantic_alignment": "yes" if i == 0 else "no",
            "complete_demo": "yes" if i == 0 else "no",
            "usable_demo": "yes" if i == 0 else "no",
            "note": "manual",
        }
        for i in range(2)
    ]
    visual = [output("i0", "demo_pass", "yes"), output("i1", "demo_pass", "no")]
    alignment = [
        output("i0", "semantic_pass", "yes"),
        {"item_id": "i1", "error": "JSONDecodeError", "result": None},
    ]
    rows, summary = evaluate(
        blind,
        semantic,
        ledger,
        {"qwen": visual},
        {"qwen": alignment},
        2,
        model_error_policy="fail_closed",
    )
    assert rows[1]["semantic_decisions"]["qwen"] is False
    assert summary["model_error_policy"] == "fail_closed"
    assert summary["model_failed_items"]["semantic"]["qwen"] == ["i1"]
