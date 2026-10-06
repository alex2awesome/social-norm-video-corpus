import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_commentary_temporal_repair_audit.py"
SPEC = importlib.util.spec_from_file_location("temporal_repair_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_first_temporal_repair_audit_is_complete_and_strict():
    run = ROOT / "audit_runs" / "20260729_commentary_temporal_repairs_v1"
    plan = (
        ROOT
        / "audit_runs"
        / "20260729_commentary_dual_exact_candidate_clips_v2"
        / "temporal_repair_plan_v1.jsonl"
    )
    result = MODULE.evaluate(
        plan,
        run / "manifest.jsonl",
        run / "manual_ledger.tsv",
    )
    assert result["candidates"] == 7
    assert result["visible_action"] == 5
    assert result["label_overlay_clean"] == 2
    assert result["bounds_clean"] == 4
    assert result["passed_post_transform"] == 1
    assert result["ready_for_next_repair"] == 5
    assert result["failed_temporal_repair"] == 1
    assert result["policy"] == "shadow_only_no_corpus_mutation"
