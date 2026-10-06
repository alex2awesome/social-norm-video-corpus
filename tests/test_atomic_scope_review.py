from __future__ import annotations

import csv
import json

import pytest

from scripts.audit_atomic_scope_review import verify


def write_fixture(tmp_path, derived="yes", item_id="x"):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"item_id": item_id, "decision": "reject", "is_social_norm": "yes"})
        + "\n"
    )
    review = tmp_path / "review.tsv"
    with review.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "item_id",
                "actor_kind",
                "behavior_kind",
                "affected_context_kind",
                "expectation_kind",
                "derived_is_social_norm",
                "manual_scope_reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "item_id": item_id,
                "actor_kind": "person",
                "behavior_kind": "speech_act",
                "affected_context_kind": "person",
                "expectation_kind": "interpersonal_treatment",
                "derived_is_social_norm": derived,
                "manual_scope_reason": "Grounded interpersonal speech act.",
            }
        )
    return source, review


def test_complete_review_is_verified(tmp_path):
    source, review = write_fixture(tmp_path)
    summary = verify(source, review)
    assert summary["reviewed"] == 1
    assert summary["old_to_derived_scope"] == {"yes->yes": 1}


def test_incorrect_hand_entered_composite_is_rejected(tmp_path):
    source, review = write_fixture(tmp_path, derived="no")
    with pytest.raises(ValueError, match="incorrect derived_is_social_norm"):
        verify(source, review)


def test_incomplete_review_is_rejected(tmp_path):
    source, review = write_fixture(tmp_path)
    source.write_text(
        source.read_text()
        + json.dumps({"item_id": "y", "decision": "reject", "is_social_norm": "no"})
        + "\n"
    )
    with pytest.raises(ValueError, match="cover every source item"):
        verify(source, review)
