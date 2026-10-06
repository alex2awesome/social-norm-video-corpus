import hashlib

import pytest

from scripts.select_instructional_v17_causal_holdout import (
    gemma_passes,
    qwen_passes,
    select,
)


def board(key: str) -> dict:
    return {
        "item_id": key,
        "sheet_path": f"/boards/{key}.jpg",
        "sheet_sha256": hashlib.sha256(key.encode()).hexdigest(),
    }


def source(key: str, uid: str) -> dict:
    return {
        "item_id": key,
        "uid": uid,
        "source_platform": "dailymotion",
        "polarity": "violation",
    }


def scored(key: str, result: dict) -> dict:
    return {"item_id": key, "error": None, "result": result}


def good_qwen() -> dict:
    return {
        "demo_usable": "uncertain",
        "demo_usable_raw": "yes",
        "social_scope": "tacit_interpersonal",
        "scene_role": "situated_scene",
        "rejection_reason": "none",
    }


def good_gemma() -> dict:
    return {
        "visual_record_supports_concrete_demo": "yes",
        "same_actor_and_target": "yes",
        "causal_or_intent_match": "not_required",
        "proposed_violation_polarity_matches": "yes",
        "quote_role": "narrated_depicted_action",
    }


def test_frozen_predicates() -> None:
    assert qwen_passes(good_qwen())
    assert gemma_passes(good_gemma())
    for key in (
        "visual_record_supports_concrete_demo",
        "same_actor_and_target",
        "proposed_violation_polarity_matches",
    ):
        broken = good_gemma()
        broken[key] = "no"
        assert not gemma_passes(broken)


def test_selects_candidate_and_first_failure_controls_blindly() -> None:
    sources = [source(f"i{i}", f"u{i}") for i in range(5)]
    boards = [board(f"i{i}") for i in range(5)]
    glm = [scored(f"i{i}", {"demo_usable": "yes"}) for i in range(5)]
    qwen = [scored(f"i{i}", good_qwen()) for i in range(5)]
    gemma = [scored(f"i{i}", good_gemma()) for i in range(5)]
    glm[1]["result"]["demo_usable"] = "no"
    qwen[2]["result"]["rejection_reason"] = "no_affected_party"
    gemma[3]["result"]["same_actor_and_target"] = "no"

    semantic, blind, summary = select(
        sources,
        boards,
        glm,
        qwen,
        gemma,
        [{"uid": "u4"}],
        candidate_limit=40,
        control_limit=15,
    )

    assert {row["band"] for row in semantic} == {
        "candidate",
        "glm_v10a_reject",
        "qwen_v16_reject",
        "gemma_causal_reject",
    }
    assert all(set(row) == {
        "audit_index", "candidate_id", "sheet_path", "sheet_sha256"
    } for row in blind)
    assert "u4" not in {row["uid"] for row in semantic}
    assert summary["eligible_source_distinct"] == 4
    assert summary["excluded_prior_uids"] == 1


def test_requires_exact_model_coverage() -> None:
    with pytest.raises(ValueError, match="gemma coverage mismatch"):
        select(
            [source("i0", "u0")],
            [board("i0")],
            [scored("i0", {"demo_usable": "yes"})],
            [scored("i0", good_qwen())],
            [],
            [],
        )


def test_rejects_duplicate_eligible_uid() -> None:
    sources = [source("i0", "u0"), source("i1", "u0")]
    boards = [board("i0"), board("i1")]
    glm = [scored(key, {"demo_usable": "yes"}) for key in ("i0", "i1")]
    qwen = [scored(key, good_qwen()) for key in ("i0", "i1")]
    gemma = [scored(key, good_gemma()) for key in ("i0", "i1")]
    with pytest.raises(ValueError, match="repeats uid"):
        select(sources, boards, glm, qwen, gemma, [])
