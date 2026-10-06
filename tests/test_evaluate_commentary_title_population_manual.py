import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_commentary_title_population_manual.py"
SPEC = importlib.util.spec_from_file_location("commentary_manual_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_population_is_complete_and_shadow_only():
    audit = ROOT / "audit_runs" / "20260728_commentary_title_population_dense_v1"
    evaluated, summary = MODULE.evaluate(
        audit / "semantic_selection.jsonl",
        audit / "blind_visual_ledger.tsv",
        audit / "post_reveal_alignment_ledger.tsv",
        audit / "post_reveal_clear_source_adjudication.json",
    )

    assert len(evaluated) == 154
    assert summary["coverage_complete"] is True
    assert summary["policy"] == "shadow_source_routing_only_no_corpus_mutation"
    assert sum(summary["route_counts"].values()) == 154
    assert summary["source_usable_candidates"] == 93
    assert summary["requires_exact_clip_audit"] == 93


def test_format_reroutes_and_mismatches_are_preserved():
    audit = ROOT / "audit_runs" / "20260728_commentary_title_population_dense_v1"
    evaluated, _ = MODULE.evaluate(
        audit / "semantic_selection.jsonl",
        audit / "blind_visual_ledger.tsv",
        audit / "post_reveal_alignment_ledger.tsv",
        audit / "post_reveal_clear_source_adjudication.json",
    )
    by_index = {row["audit_index"]: row for row in evaluated}

    for index in (33, 64, 78, 139, 147):
        assert by_index[index]["route"] == "instructional_demo_review"
    for index in (61, 70, 103, 115, 142):
        assert by_index[index]["route"] == "relabel_visual_localization_review"
    assert by_index[94]["route"] == "reject_no_visible_social_event"
