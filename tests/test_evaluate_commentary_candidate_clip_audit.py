import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_commentary_candidate_clip_audit.py"
SPEC = importlib.util.spec_from_file_location("candidate_clip_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_candidate_clip_audit_is_complete():
    source = ROOT / "audit_runs" / "20260728_commentary_title_population_dense_v1"
    audit = ROOT / "audit_runs" / "20260729_commentary_dual_exact_candidate_clips_v2"
    result = MODULE.evaluate(
        source / "dual_exact_candidate_clip_plan.jsonl",
        audit / "manifest.jsonl",
        audit / "failures.jsonl",
        audit / "post_transform_manual_ledger.tsv",
    )
    assert result["planned"] == 32
    assert result["rendered"] == 30
    assert result["render_failed"] == 2
    assert result["passed_first_transform"] == 3
    assert result["clean_commentary_candidates"] == 2
    assert result["clean_instructional_reroutes"] == 1
    assert result["repair_required"] == 26
    assert result["source_false_positive"] == 1
    assert result["policy"] == "shadow_only_no_corpus_mutation"
