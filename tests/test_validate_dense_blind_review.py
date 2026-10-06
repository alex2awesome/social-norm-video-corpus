import json
from pathlib import Path

import pytest

from scripts.validate_dense_blind_review import validate_dense_review


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def valid_review(item_id: str, audit_index: int) -> dict:
    return {
        "item_id": item_id,
        "audit_index": audit_index,
        "dense_scene_visible": "yes",
        "dense_concrete_action_visible": "uncertain",
        "temporal_description": "Two people interact across the sampled interval.",
        "sparse_changed": "no",
        "review_complete": "yes",
    }


def sparse_review(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "situated_social_scene": "yes",
        "concrete_behavior_visible": "uncertain",
        "dense_review_required": "yes",
    }


def test_exact_dense_review_passes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    sparse = tmp_path / "sparse.jsonl"
    review = tmp_path / "review.jsonl"
    write_jsonl(
        manifest,
        [
            {"item_id": "a", "audit_index": 2},
            {"item_id": "b", "audit_index": 7},
        ],
    )
    write_jsonl(sparse, [sparse_review("a"), sparse_review("b")])
    write_jsonl(review, [valid_review("b", 7), valid_review("a", 2)])
    result = validate_dense_review(manifest, sparse, review)
    assert result["coverage_exact"] is True
    assert result["review_items"] == 2
    assert result["sparse_crosscheck_valid"] is True


@pytest.mark.parametrize(
    "mutator",
    [
        lambda rows: rows[:1],
        lambda rows: rows + [valid_review("a", 2)],
        lambda rows: [{**rows[0], "audit_index": 99}, rows[1]],
        lambda rows: [{**rows[0], "dense_scene_visible": "maybe"}, rows[1]],
        lambda rows: [{**rows[0], "temporal_description": " "}, rows[1]],
        lambda rows: [{**rows[0], "review_complete": "no"}, rows[1]],
        lambda rows: [{**rows[0], "unexpected": True}, rows[1]],
    ],
)
def test_dense_review_fails_closed(tmp_path: Path, mutator) -> None:
    manifest = tmp_path / "manifest.jsonl"
    sparse = tmp_path / "sparse.jsonl"
    review = tmp_path / "review.jsonl"
    write_jsonl(
        manifest,
        [
            {"item_id": "a", "audit_index": 2},
            {"item_id": "b", "audit_index": 7},
        ],
    )
    write_jsonl(sparse, [sparse_review("a"), sparse_review("b")])
    write_jsonl(review, mutator([valid_review("a", 2), valid_review("b", 7)]))
    with pytest.raises(ValueError):
        validate_dense_review(manifest, sparse, review)


def test_sparse_change_flag_is_crosschecked(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    sparse = tmp_path / "sparse.jsonl"
    review = tmp_path / "review.jsonl"
    write_jsonl(manifest, [{"item_id": "a", "audit_index": 2}])
    write_jsonl(sparse, [sparse_review("a")])
    row = valid_review("a", 2)
    row["dense_concrete_action_visible"] = "yes"
    write_jsonl(review, [row])
    with pytest.raises(ValueError, match="sparse_changed"):
        validate_dense_review(manifest, sparse, review)
