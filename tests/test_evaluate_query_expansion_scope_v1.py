from pathlib import Path

from scripts.evaluate_query_expansion_scope_v1 import evaluate


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_query_scope_audit_passes() -> None:
    report = evaluate(
        ROOT / "config/typical_social_norm_queries_v1.yaml",
        ROOT / "audit_runs/20260807_query_expansion_scope_v1/manual_negative_scope_audit.jsonl",
    )
    assert report["items"] == 90
    assert report["false_allow"] == 0
    assert report["false_block"] == 0
    assert report["promotion_gate"]["passed"] is True
