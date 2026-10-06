from scripts.evaluate_instructional_v9_discovery import (
    metric,
    original_violation_usable,
    wilson_95,
)


def test_metric_reports_selected_errors_and_recall():
    rows = [
        {
            "manual": {
                "audit_index": 0,
                "usable_after_relabel": True,
            },
            "selected": True,
        },
        {
            "manual": {
                "audit_index": 1,
                "usable_after_relabel": False,
            },
            "selected": True,
        },
        {
            "manual": {
                "audit_index": 2,
                "usable_after_relabel": True,
            },
            "selected": False,
        },
    ]
    result = metric(
        rows,
        lambda row: row["selected"],
        "usable_after_relabel",
    )
    assert result["selected"] == 2
    assert result["positive"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["false_positive_audit_indices"] == [1]
    assert result["false_negative_audit_indices"] == [2]


def test_wilson_interval_is_bounded():
    low, high = wilson_95(9, 10)
    assert 0 < low < 0.9 < high <= 1


def test_original_violation_rule_uses_frozen_metadata_polarity():
    row = {
        "manifest": {"polarity": "violation"},
        "alignments": {"qwen32b": {"usable_after_relabel": "yes"}},
    }
    assert original_violation_usable(row, "qwen32b")
    row["manifest"]["polarity"] = "correct"
    assert not original_violation_usable(row, "qwen32b")
    row["manifest"]["polarity"] = "violation"
    row["alignments"]["qwen32b"]["usable_after_relabel"] = "no"
    assert not original_violation_usable(row, "qwen32b")
