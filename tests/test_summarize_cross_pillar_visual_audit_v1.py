from pathlib import Path

from scripts.summarize_cross_pillar_visual_audit_v1 import summarize


def test_frozen_large_audit_has_complete_balanced_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    report = summarize(root / "audit_runs/20260807_cross_pillar_supervision_v1/manual_visual_audit.jsonl")
    assert report["items"] == 60
    assert {k: v["items"] for k, v in report["pillars"].items()} == {
        "commentary": 20, "instructional": 20, "witnessed": 20}
    assert report["pillars"]["instructional"]["decisions"] == {"no": 14, "uncertain": 3, "yes": 3}
    assert report["pillars"]["witnessed"]["decisions"] == {"no": 15, "uncertain": 4, "yes": 1}
    assert report["pillars"]["commentary"]["decisions"] == {"no": 11, "uncertain": 2, "yes": 7}
