import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_witnessed_search_shadow", ROOT / "scripts/run_witnessed_search_shadow.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_sha256_json_is_key_order_invariant():
    assert MODULE.sha256_json({"a": 1, "b": 2}) == MODULE.sha256_json({"b": 2, "a": 1})


def test_shadow_is_inactive_and_final_gate_is_video_review():
    import yaml

    cfg = yaml.safe_load((ROOT / "config/witnessed_search_shadow_v1.yaml").read_text())
    assert cfg["active"] is False
    contract = cfg["selection_contract"]
    assert contract["state_database_mode"] == "read_only"
    assert contract["title_or_query_can_never_accept"] is True
    assert contract["final_gate"] == "full_witnessed_v1_video_review"


def test_v2_shadow_is_read_only_and_requires_full_video_review():
    import yaml

    cfg = yaml.safe_load((ROOT / "config/witnessed_search_shadow_v2.yaml").read_text())
    assert cfg["active"] is False
    assert cfg["version"] == "witnessed_search_shadow_v2"
    contract = cfg["selection_contract"]
    assert contract["state_database_mode"] == "read_only"
    assert contract["title_or_query_can_never_accept"] is True
    assert contract["final_gate"] == "full_witnessed_video_review_and_exact_splice_audit"
    assert contract["production_eligible_without_full_rank_sample_review"] is False
    result = cfg["audit_result_20260721"]
    assert result["production_eligible"] is False
    assert result["strict_accept_after_exact_splice"] == 1
    assert result["require_source_disjoint_replication_before_any_promotion"] is True


def test_instructional_search_shadow_is_inactive_and_demo_gated():
    import yaml

    cfg = yaml.safe_load((ROOT / "config/instructional_search_shadow_v1.yaml").read_text())
    assert cfg["active"] is False
    assert cfg["version"] == "instructional_search_shadow_v1"
    contract = cfg["selection_contract"]
    assert contract["state_database_mode"] == "read_only"
    assert contract["title_or_query_can_never_accept"] is True
    assert contract["final_gate"] == "full_instructional_v4_video_review"
    assert contract["require_exact_demo_inside_the_accepted_clip"] is True
    assert contract["any_medium_allowed"] is True
