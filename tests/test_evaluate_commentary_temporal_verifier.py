import pytest

from scripts.evaluate_commentary_temporal_verifier import evaluate


def test_evaluate_counts_errors_as_fail_closed_predictions():
    gold = [
        {"audit_index": 1, "manual_gold": "positive"},
        {"audit_index": 2, "manual_gold": "negative"},
        {"audit_index": 3, "manual_gold": "positive"},
        {"audit_index": 4, "manual_gold": "negative"},
    ]
    outputs = [
        {"audit_index": 1, "strict_pass": True, "error": None},
        {"audit_index": 2, "strict_pass": True, "error": None},
        {"audit_index": 3, "strict_pass": True, "error": "timeout"},
        {"audit_index": 4, "strict_pass": False, "error": None},
    ]
    summary, disagreements = evaluate(
        gold,
        outputs,
        minimum_precision=0.5,
        minimum_recall=0.5,
    )
    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 1
    assert summary["tn"] == 1
    assert summary["errors"] == 1
    assert not summary["candidate_for_disjoint_holdout"]
    assert [row["audit_index"] for row in disagreements] == [2, 3]


def test_evaluate_requires_exact_index_coverage():
    with pytest.raises(ValueError, match="index mismatch"):
        evaluate(
            [{"audit_index": 1, "manual_gold": "positive"}],
            [],
        )


def test_evaluate_marks_threshold_passing_candidate_for_holdout():
    gold = [
        {"audit_index": 1, "manual_gold": "positive"},
        {"audit_index": 2, "manual_gold": "negative"},
    ]
    outputs = [
        {"audit_index": 1, "strict_pass": True, "error": None},
        {"audit_index": 2, "strict_pass": False, "error": None},
    ]
    summary, disagreements = evaluate(gold, outputs)
    assert summary["candidate_for_disjoint_holdout"]
    assert disagreements == []
