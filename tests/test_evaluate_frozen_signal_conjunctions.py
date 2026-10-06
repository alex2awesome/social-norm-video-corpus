from scripts.evaluate_frozen_signal_conjunctions import rule_metrics


def test_rule_metrics_lists_every_selected_and_error_item():
    result = rule_metrics(
        ["a", "b", "c", "d"],
        {"a": 1, "b": 0, "c": 1, "d": 0},
        {"a": True, "b": True, "c": False, "d": False},
    )
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["selected_item_ids"] == ["a", "b"]
    assert result["false_positive_item_ids"] == ["b"]
    assert result["false_negative_item_ids"] == ["c"]
