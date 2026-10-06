from scripts.evaluate_witnessed_strict_holdout import (
    binary_metrics,
    collapse_successful,
)


def test_collapse_successful_preserves_failed_attempt_and_uses_success():
    rows = [
        {"item_id": "a", "error": "bad", "result": None},
        {"item_id": "a", "error": None, "result": {"ok": 1}},
        {"item_id": "b", "error": None, "result": {"ok": 2}},
    ]
    successful, failures = collapse_successful(rows)
    assert failures == 1
    assert successful["a"]["result"] == {"ok": 1}
    assert successful["b"]["result"] == {"ok": 2}


def test_binary_metrics_reports_selected_items_and_errors():
    report = binary_metrics(
        {"a", "b", "c", "d"},
        {"a", "b"},
        {"a", "c"},
    )
    assert report["tp"] == 1
    assert report["fp"] == 1
    assert report["fn"] == 1
    assert report["tn"] == 1
    assert report["precision"] == 0.5
    assert report["recall"] == 0.5
    assert report["true_positive_items"] == ["a"]
    assert report["false_positive_items"] == ["c"]
