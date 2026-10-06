import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_instructional_v8_fresh_validation.py"
SPEC = importlib.util.spec_from_file_location("evaluate_v8_fresh", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_wilson_is_bounded_and_contains_rate():
    low, high = MODULE.wilson_95(43, 60)
    assert 0 <= low < 43 / 60 < high <= 1


def test_real_frozen_audit_fails_promotion_and_recovers_rejects():
    audit = ROOT / "audit_runs" / "20260728_instructional_v8_fresh_validation_v1"
    joined, summary = MODULE.evaluate(
        audit / "validation_manifest.jsonl",
        audit / "dense36" / "blind_dense_adjudication.jsonl",
        audit / "dense36" / "semantic_adjudication.json",
    )
    assert len(joined) == 90
    assert summary["promotion_pass"] is False
    assert (
        summary["bands"]["v8_dual_strict"]["strict_situated_event"]["positive"]
        == 43
    )
    assert (
        summary["bands"]["v8_conjunction_reject"]["strict_situated_event"]["positive"]
        == 19
    )
    assert (
        summary["bands"]["all_90"]["usable_after_relabel"]["positive"]
        == 65
    )


def test_semantic_override_counts_are_explicit():
    audit = ROOT / "audit_runs" / "20260728_instructional_v8_fresh_validation_v1"
    _, summary = MODULE.evaluate(
        audit / "validation_manifest.jsonl",
        audit / "dense36" / "blind_dense_adjudication.jsonl",
        audit / "dense36" / "semantic_adjudication.json",
    )
    assert summary["semantic_status_counts"] == {
        "exact": 58,
        "unusable": 25,
        "usable_after_relabel": 7,
    }
