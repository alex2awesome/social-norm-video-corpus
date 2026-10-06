import json
from pathlib import Path

import pytest

from scripts.validate_instructional_post_reveal_review import final_visual_demo, validate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def row(item_id: str = "instructional:u:0") -> dict:
    return {
        "item_id": item_id,
        "audit_index": 0,
        "visual_demo_present": "yes",
        "assigned_norm_is_social_norm": "yes",
        "demo_visually_matches_assigned_norm": "yes",
        "polarity_matches_depiction": "yes",
        "quote_grounded_in_interval": "yes",
        "clip_is_demo_dominant": "yes",
        "strict_instructional_pass": "yes",
        "recoverable_route": "instructional",
        "required_repair": "none",
        "evidence_note": "A localized interpersonal demonstration matches the label.",
        "review_complete": "yes",
    }


def fixture(
    tmp_path: Path,
    review: dict | None = None,
    *,
    dense_required: bool = False,
):
    packets = tmp_path / "packets.jsonl"
    sparse = tmp_path / "sparse.jsonl"
    dense = tmp_path / "dense.jsonl"
    reviews = tmp_path / "review.jsonl"
    write_jsonl(
        packets,
        [{"item_id": "instructional:u:0", "audit_index": 0}],
    )
    write_jsonl(
        sparse,
        [
            {
                "item_id": "instructional:u:0",
                "situated_social_scene": "yes",
                "concrete_behavior_visible": "yes",
                "dense_review_required": "yes" if dense_required else "no",
            }
        ],
    )
    write_jsonl(
        dense,
        (
            [
                {
                    "item_id": "instructional:u:0",
                    "dense_scene_visible": "yes",
                    "dense_concrete_action_visible": "yes",
                    "review_complete": "yes",
                }
            ]
            if dense_required
            else []
        ),
    )
    write_jsonl(reviews, [review or row()])
    return packets, sparse, dense, reviews


def test_valid_strict_pass(tmp_path: Path) -> None:
    result = validate(*fixture(tmp_path))
    assert result["strict_instructional_passes"] == 1
    assert result["visual_demos"] == 1


def test_dense_review_can_supply_final_visual_label(tmp_path: Path) -> None:
    result = validate(*fixture(tmp_path, dense_required=True))
    assert result["strict_instructional_passes"] == 1


def test_strict_pass_requires_every_gate(tmp_path: Path) -> None:
    review = row()
    review["clip_is_demo_dominant"] = "uncertain"
    with pytest.raises(ValueError, match="strict pass is inconsistent"):
        validate(*fixture(tmp_path, review))


def test_visual_label_cannot_change_after_reveal(tmp_path: Path) -> None:
    review = row()
    review["visual_demo_present"] = "no"
    review["demo_visually_matches_assigned_norm"] = "no"
    review["strict_instructional_pass"] = "no"
    with pytest.raises(ValueError, match="contradicts frozen blind review"):
        validate(*fixture(tmp_path, review))


def test_absent_demo_cannot_match_norm(tmp_path: Path) -> None:
    review = row()
    review["visual_demo_present"] = "no"
    review["strict_instructional_pass"] = "no"
    packets, sparse, dense, reviews = fixture(tmp_path, review)
    sparse_row = json.loads(sparse.read_text())
    sparse_row["situated_social_scene"] = "no"
    sparse_row["concrete_behavior_visible"] = "no"
    write_jsonl(sparse, [sparse_row])
    with pytest.raises(ValueError, match="absent demo cannot visually match"):
        validate(packets, sparse, dense, reviews)


def test_dense_coverage_must_follow_sparse_flag(tmp_path: Path) -> None:
    packets, sparse, dense, reviews = fixture(tmp_path, dense_required=True)
    dense.write_text("")
    with pytest.raises(ValueError, match="dense review coverage"):
        validate(packets, sparse, dense, reviews)


def test_review_coverage_must_be_exact(tmp_path: Path) -> None:
    packets, sparse, dense, reviews = fixture(tmp_path)
    write_jsonl(reviews, [row("instructional:other:0")])
    with pytest.raises(ValueError, match="order/coverage"):
        validate(packets, sparse, dense, reviews)


def test_final_visual_demo_supports_rank_confirmation_schema() -> None:
    item_id = "instructional:u:0"
    sparse = {
        "item_id": item_id,
        "visual_demo_present": "no",
        "dense_followup": "yes",
    }
    dense = {
        item_id: {
            "item_id": item_id,
            "visual_demo_present": "yes",
            "review_complete": "yes",
        }
    }
    assert final_visual_demo(item_id, sparse, dense) is True
