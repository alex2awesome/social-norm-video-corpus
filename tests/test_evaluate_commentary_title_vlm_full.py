import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_commentary_title_vlm_full.py"
SPEC = importlib.util.spec_from_file_location("commentary_vlm_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_full_vlm_evaluation_covers_frozen_population():
    audit = ROOT / "audit_runs" / "20260728_commentary_title_population_dense_v1"
    output = MODULE.evaluate(
        audit / "manual_source_routes.jsonl",
        audit / "qwen_labelblind_154.jsonl",
        audit / "qwen_title_retrieval_v1.jsonl",
        audit / "gemma_title_retrieval_v1.jsonl",
    )
    assert output["gold_counts"]["source_usable_any_route"] == 93
    assert output["gold_counts"]["exact_title_event"] == 74
    assert output["slices"]["all_154"]["items"] == 154
    assert output["slices"]["clear_94_validation"]["items"] == 94
    assert output["slices"]["ambiguous_60_development"]["items"] == 60
    assert output["policy"] == "shadow_ranking_only_no_automatic_acceptance"
    assert (
        output["slices"]["ambiguous_60_development"]["rules"][
            "dual_retrieval_core"
        ]["usable_any_route"]["tp"]
        > 0
    )


def test_every_rule_has_both_gold_contracts():
    audit = ROOT / "audit_runs" / "20260728_commentary_title_population_dense_v1"
    output = MODULE.evaluate(
        audit / "manual_source_routes.jsonl",
        audit / "qwen_labelblind_154.jsonl",
        audit / "qwen_title_retrieval_v1.jsonl",
        audit / "gemma_title_retrieval_v1.jsonl",
    )
    for slice_result in output["slices"].values():
        for result in slice_result["rules"].values():
            assert set(result) == {"usable_any_route", "exact_title_event"}
