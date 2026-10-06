import json
from pathlib import Path

import pytest

from scripts.validate_commentary_post_reveal_review import final_visual_event, validate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def review(item_id: str = "commentary:u:0") -> dict:
    return {
        "item_id": item_id,
        "audit_index": 0,
        "visual_event_present": "yes",
        "assigned_norm_is_social_norm": "yes",
        "statement_is_normative_evidence": "yes",
        "quote_grounded_in_transcript": "yes",
        "visible_event_matches_statement": "yes",
        "polarity_matches_visible_event": "yes",
        "visual_window_is_event_dominant": "yes",
        "strict_commentary_visual_pass": "yes",
        "recoverable_route": "commentary_visual",
        "required_repair": "none",
        "evidence_note": "Visible theft and aligned condemnation.",
        "review_complete": "yes",
    }


def fixture(tmp_path: Path, *, dense: bool = True) -> tuple[Path, Path, Path, Path]:
    item_id = "commentary:u:0"
    packets = tmp_path / "packets.jsonl"
    sparse = tmp_path / "sparse.jsonl"
    dense_path = tmp_path / "dense.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    write_jsonl(packets, [{"item_id": item_id, "audit_index": 0}])
    write_jsonl(
        sparse,
        [
            {
                "item_id": item_id,
                "dense_review_required": "yes" if dense else "no",
                "situated_social_scene": "yes",
                "concrete_behavior_visible": "yes",
            }
        ],
    )
    write_jsonl(
        dense_path,
        (
            [
                {
                    "item_id": item_id,
                    "dense_scene_visible": "yes",
                    "dense_concrete_action_visible": "yes",
                    "review_complete": "yes",
                }
            ]
            if dense
            else []
        ),
    )
    write_jsonl(reviews, [review()])
    return packets, sparse, dense_path, reviews


def test_valid_strict_review(tmp_path: Path) -> None:
    result = validate(*fixture(tmp_path))
    assert result["strict_commentary_visual_passes"] == 1
    assert result["visual_events"]["yes"] == 1


def test_visual_judgment_cannot_change_after_reveal(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    row = review()
    row["visual_event_present"] = "no"
    row["strict_commentary_visual_pass"] = "no"
    write_jsonl(paths[-1], [row])
    with pytest.raises(ValueError, match="contradicts frozen"):
        validate(*paths)


def test_absent_visual_cannot_match_statement(tmp_path: Path) -> None:
    paths = fixture(tmp_path, dense=False)
    sparse = {
        "item_id": "commentary:u:0",
        "dense_review_required": "no",
        "situated_social_scene": "no",
        "concrete_behavior_visible": "no",
    }
    write_jsonl(paths[1], [sparse])
    row = review()
    row["visual_event_present"] = "no"
    row["strict_commentary_visual_pass"] = "no"
    write_jsonl(paths[-1], [row])
    with pytest.raises(ValueError, match="cannot visibly match"):
        validate(*paths)


def test_strict_pass_must_match_all_gates(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    row = review()
    row["polarity_matches_visible_event"] = "no"
    write_jsonl(paths[-1], [row])
    with pytest.raises(ValueError, match="strict pass is inconsistent"):
        validate(*paths)


def test_review_order_and_coverage_are_exact(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    write_jsonl(paths[-1], [])
    with pytest.raises(ValueError, match="order/coverage"):
        validate(*paths)


def test_final_visual_event_supports_statement_centered_schema() -> None:
    assert (
        final_visual_event(
            "commentary:u:0",
            {
                "localized_human_event_visible": "yes",
                "localized_social_interaction_visible": "yes",
            },
            {},
        )
        == "yes"
    )
    assert (
        final_visual_event(
            "commentary:u:0",
            {
                "localized_human_event_visible": "yes",
                "localized_social_interaction_visible": "uncertain",
            },
            {},
        )
        == "uncertain"
    )
