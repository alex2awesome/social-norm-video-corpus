import pytest

from scripts.select_instructional_v22_source_disjoint_holdout import (
    blind_record,
    promotion_gate,
    select,
    stratum,
)


def row(index: int, category: str, polarity: str, uid: str | None = None) -> dict:
    return {
        "item_id": f"instructional:{uid or f'u{index}'}:{index}",
        "uid": uid or f"u{index}",
        "category": category,
        "polarity": polarity,
        "source_clip": f"/{index}.mp4",
        "norm": "sealed semantic",
    }


def test_select_is_deterministic_balanced_and_source_disjoint() -> None:
    rows = [
        row(0, "a", "correct", "shared"),
        row(1, "a", "correct", "shared"),
        row(2, "a", "violation"),
        row(3, "b", "correct"),
        row(4, "b", "violation"),
        row(5, "c", "explanation"),
    ]
    first = select(rows, 4, "seed", {"u5"})
    second = select(rows, 4, "seed", {"u5"})
    assert [r["item_id"] for r in first] == [r["item_id"] for r in second]
    assert len({r["uid"] for r in first}) == 4
    assert "u5" not in {r["uid"] for r in first}
    assert len({stratum(r) for r in first}) >= 3


def test_select_rejects_insufficient_unique_sources() -> None:
    rows = [row(0, "a", "correct", "same"), row(1, "b", "violation", "same")]
    with pytest.raises(ValueError, match="insufficient source-disjoint"):
        select(rows, 2, "seed", set())


def test_blind_record_contains_no_semantic_label() -> None:
    record = blind_record(row(0, "a", "correct"), 7)
    assert record["candidate_id"] == "v22holdout-0007"
    assert "norm" not in record
    assert "category" not in record
    assert "polarity" not in record


def test_intensive_gate_is_stricter_and_requires_100() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        promotion_gate(99, "intensive")
    gate = promotion_gate(120, "intensive")
    assert gate["dual_model_pipeline_min_precision"] == 0.85
    assert gate["source_cluster_exact_min_precision"] == 0.85
    assert gate["acceptance_or_rejection_authority"] is False


def test_title_v23_union_gate_matches_preregistered_target() -> None:
    gate = promotion_gate(100, "intensive", "title_v23_union")
    assert gate["title_or_v23_min_precision"] == 0.75
    assert gate["title_or_v23_min_recall"] == 0.65


def test_blind_episode_relaxed_gate_is_visual_routing_only() -> None:
    gate = promotion_gate(100, "intensive", "blind_episode_relaxed")
    assert gate["qwen_blind_episode_without_completeness_min_precision"] == 0.80
    assert gate["qwen_blind_episode_without_completeness_min_recall"] == 0.40
    assert gate["qwen_blind_episode_without_completeness_min_selected"] == 15
    assert gate["exact_label_symbolic_result_is_exploratory"] is True
    assert gate["acceptance_or_rejection_authority"] is False


def test_demo_consensus_v2_requires_fresh_and_post_scale_transfer() -> None:
    gate = promotion_gate(100, "intensive", "demo_consensus_v2")
    assert gate["qwen_v1_v2_consensus_min_precision"] == 0.80
    assert gate["qwen_v1_v2_consensus_min_recall"] == 0.40
    assert gate["qwen_v1_v2_consensus_min_selected"] == 15
    assert gate["post_scale_transfer_audit_required"] is True
    assert gate["uniform_source_disjoint_sample"] is True
    assert gate["acceptance_or_rejection_authority"] is False
    assert gate["uniform_source_disjoint_sample"] is True
    assert gate["acceptance_or_rejection_authority"] is False
