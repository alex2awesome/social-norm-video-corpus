from pathlib import Path

from scripts.evaluate_related_snowball_gate_v1 import evaluate


def test_frozen_parent_audit_passes_without_false_propagation() -> None:
    root = Path(__file__).resolve().parents[1]
    report = evaluate(root / "audit_runs/20260808_related_snowball_gate_v1/manual_parent_audit.jsonl")
    assert report["items"] == 20
    assert report["true_allow"] == 1
    assert report["true_block"] == 19
    assert report["false_allow"] == 0
    assert report["false_block"] == 0
    assert report["promotion_gate"]["passed"] is True
