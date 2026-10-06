import json
from pathlib import Path

import pytest

from scripts.validate_witnessed_post_reveal_review import validate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def row(item_id: str = "witnessed:u:0") -> dict:
    return {
        "item_id": item_id,
        "audit_index": 0,
        "literal_social_norm_violation_visible": "yes",
        "assigned_norm_matches_event": "yes",
        "reaction_is_organic_bystander_disapproval": "yes",
        "violation_precedes_reaction": "yes",
        "reaction_spliceable_without_label_leak": "yes",
        "strict_witnessed_pass": "yes",
        "recoverable_route": "witnessed",
        "required_repair": "none",
        "evidence_note": "The action and subsequent response are both visible.",
        "review_complete": "yes",
    }


def fixture(tmp_path: Path, review: dict | None = None):
    packets = tmp_path / "packets.jsonl"
    sparse = tmp_path / "sparse.jsonl"
    reviews = tmp_path / "review.jsonl"
    write_jsonl(packets, [{"item_id": "witnessed:u:0", "audit_index": 0}])
    write_jsonl(sparse, [{"item_id": "witnessed:u:0"}])
    write_jsonl(reviews, [review or row()])
    return packets, sparse, reviews


def test_valid_strict_pass(tmp_path: Path) -> None:
    result = validate(*fixture(tmp_path))
    assert result["strict_witnessed_passes"] == 1
    assert result["strict_witnessed_rate"] == 1.0


def test_strict_pass_requires_every_evidence_field(tmp_path: Path) -> None:
    review = row()
    review["reaction_is_organic_bystander_disapproval"] = "uncertain"
    with pytest.raises(ValueError, match="strict pass is inconsistent"):
        validate(*fixture(tmp_path, review))


def test_non_witnessed_route_cannot_strictly_pass(tmp_path: Path) -> None:
    review = row()
    review["recoverable_route"] = "instructional"
    with pytest.raises(ValueError, match="strict pass is inconsistent"):
        validate(*fixture(tmp_path, review))


def test_coverage_must_be_exact(tmp_path: Path) -> None:
    packets, sparse, reviews = fixture(tmp_path)
    write_jsonl(reviews, [row("witnessed:other:0")])
    with pytest.raises(ValueError, match="order/coverage"):
        validate(packets, sparse, reviews)
