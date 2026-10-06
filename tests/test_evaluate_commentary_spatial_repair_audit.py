import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_commentary_spatial_repair_audit.py"
SPEC = importlib.util.spec_from_file_location("spatial_repair_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_spatial_repair_audit_is_complete():
    audit = (
        ROOT
        / "audit_runs"
        / "20260729_commentary_dual_exact_candidate_clips_v2"
    )
    result = MODULE.evaluate(
        audit / "spatial_repair_plan_v1.jsonl",
        audit / "spatial_repair_v1" / "blind_manifest.jsonl",
        audit / "spatial_repair_v1" / "manual_ledger.tsv",
    )
    assert result["variants"] == 11
    assert result["visible_action"] == 7
    assert result["label_overlay_clean"] == 5
    assert result["ready_for_second_transform"] == 7
    assert result["failed_spatial_repair"] == 4
    assert result["passed_post_transform"] == 0
    assert result["policy"] == "shadow_only_no_corpus_mutation"


def test_materialized_second_transform_has_five_manual_passes():
    audit = (
        ROOT
        / "audit_runs"
        / "20260729_commentary_dual_exact_candidate_clips_v2"
    )
    result = MODULE.evaluate(
        audit / "spatial_repair_plan_v2.jsonl",
        audit / "spatial_repair_materialized_v2" / "manifest.jsonl",
        audit / "spatial_repair_materialized_v2" / "manual_ledger.tsv",
    )
    assert result["variants"] == 7
    assert result["visible_action"] == 6
    assert result["label_overlay_clean"] == 6
    assert result["passed_post_transform"] == 5
    assert result["ready_for_second_transform"] == 1
    assert result["failed_spatial_repair"] == 1


def test_full_frame_reaudit_preserves_five_narrowly_labeled_passes():
    audit = (
        ROOT
        / "audit_runs"
        / "20260729_commentary_dual_exact_candidate_clips_v2"
    )
    result = MODULE.evaluate(
        audit / "spatial_repair_plan_v2.jsonl",
        audit
        / "spatial_repair_materialized_v2_fullframes_reaudit"
        / "manifest.jsonl",
        audit
        / "spatial_repair_materialized_v2_fullframes_reaudit"
        / "manual_ledger.tsv",
    )
    assert result["variants"] == 7
    assert result["visible_action"] == 6
    assert result["label_overlay_clean"] == 6
    assert result["passed_post_transform"] == 5
    assert result["ready_for_second_transform"] == 1
    assert result["failed_spatial_repair"] == 1


def test_full_frame_third_transform_is_clean_after_conservative_relabel():
    audit = (
        ROOT
        / "audit_runs"
        / "20260729_commentary_dual_exact_candidate_clips_v2"
    )
    result = MODULE.evaluate(
        audit / "spatial_repair_plan_v3.jsonl",
        audit
        / "spatial_repair_materialized_v3_fullframes_reaudit"
        / "manifest.jsonl",
        audit
        / "spatial_repair_materialized_v3_fullframes_reaudit"
        / "manual_ledger.tsv",
    )
    assert result["variants"] == 1
    assert result["visible_action"] == 1
    assert result["label_overlay_clean"] == 1
    assert result["passed_post_transform"] == 1
    assert result["ready_for_second_transform"] == 0
    assert result["failed_spatial_repair"] == 0
