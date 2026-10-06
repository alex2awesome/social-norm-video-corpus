import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_instructional_v10_discovery.py"
SPEC = importlib.util.spec_from_file_location("evaluate_v10", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def real_evaluation():
    v9 = ROOT / "audit_runs" / "20260728_instructional_v9_corpus_validation_v1"
    v10 = ROOT / "audit_runs" / "20260728_instructional_v10_discovery_v1"
    return MODULE.evaluate(
        v9 / "validation_selection.jsonl",
        v9 / "blind" / "manual_visual_ledger.tsv",
        v9 / "manual_semantic_ledger.tsv",
        v10 / "glm_v10a.jsonl",
        v10 / "qwen32b_v10b.jsonl",
        v10 / "qwen32b_v10c.jsonl",
    )


def test_real_discovery_has_complete_coverage_and_is_not_promotable():
    joined, summary = real_evaluation()
    assert len(joined) == 166
    assert set(summary["coverage"].values()) == {166}
    assert summary["promotion_eligible"] is False


def test_primary_consensus_metrics_are_frozen():
    _, summary = real_evaluation()
    primary = summary["metrics"]["primary_violation_all"]
    assert primary["v10c_strict"]["exact_original"]["selected"] == 29
    assert primary["v10c_strict"]["exact_original"]["true_positive"] == 26
    assert primary["v10c_strict"]["visual_demo"]["true_positive"] == 27
    assert primary["v10c_strict"]["exact_original"]["false_positive_indices"] == [
        8,
        30,
        76,
    ]


def test_platform_ablation_is_reported_but_stays_discovery_only():
    _, summary = real_evaluation()
    metric = summary["metrics"]["primary_violation_all"]["v10c_strict_youtube"][
        "exact_original"
    ]
    assert metric["selected"] == 26
    assert metric["true_positive"] == 25
    assert summary["promotion_eligible"] is False
