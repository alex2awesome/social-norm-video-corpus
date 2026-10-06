from pathlib import Path

from scripts.evaluate_related_snowball_transfer_v1 import evaluate


def test_transfer_failure_requires_quarantine_not_parent_gate_promotion() -> None:
    root = Path(__file__).resolve().parents[1]
    report = evaluate(root / "audit_runs/20260808_related_snowball_gate_v1/transfer_audit/manual_child_audit.jsonl")
    assert report["items"] == 40
    assert report["bands"]["allow"]["decisions"] == {"no": 19, "uncertain": 1}
    assert report["bands"]["block"]["decisions"] == {"no": 19, "uncertain": 1}
    assert report["parent_gate_transfer_passed"] is False
    assert report["recommended_action"] == "quarantine_dailymotion_related_queries"
