import json

import pytest

from scripts.consolidate_manual_gold import consolidate


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_matching_repeated_decisions_preserve_provenance(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl(first, [{"item_id": "a", "decision": "accept", "evidence": "one"}])
    write_jsonl(second, [{"item_id": "a", "decision": "accept", "evidence": "two"}])

    rows = consolidate([first, second])

    assert rows[0]["decision"] == "accept"
    assert rows[0]["adjudicated"] is False
    assert len(rows[0]["source_reviews"]) == 2


def test_conflict_fails_without_explicit_adjudication(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl(first, [{"item_id": "a", "decision": "accept"}])
    write_jsonl(second, [{"item_id": "a", "decision": "reject"}])

    with pytest.raises(ValueError, match="require adjudication"):
        consolidate([first, second])


def test_adjudication_resolves_conflict_and_requires_reason(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    adjudications = tmp_path / "adjudications.jsonl"
    write_jsonl(first, [{"item_id": "a", "decision": "accept"}])
    write_jsonl(second, [{"item_id": "a", "decision": "reject"}])
    write_jsonl(
        adjudications,
        [
            {
                "item_id": "a",
                "decision": "reject",
                "adjudication_reason": "The target action is absent.",
            }
        ],
    )

    rows = consolidate([first, second], adjudications)

    assert rows[0]["decision"] == "reject"
    assert rows[0]["adjudicated"] is True
    assert {row["decision"] for row in rows[0]["source_reviews"]} == {
        "accept",
        "reject",
    }
