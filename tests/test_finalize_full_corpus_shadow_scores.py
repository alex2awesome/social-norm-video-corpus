import pytest

from scripts.finalize_full_corpus_shadow_scores import merge_scores


def score(item_id, value=None, error=None):
    return {
        "item_id": item_id,
        "low_level": {} if value is None else {"motion_mean": value},
        "error": error,
    }


def test_merge_preserves_manifest_order_and_successful_retry_wins():
    manifest = [{"item_id": "a"}, {"item_id": "b"}, {"item_id": "c"}]
    merged, summary = merge_scores(
        manifest,
        [
            [score("a", 1), score("b", error="decode")],
            [score("b", 2), score("c", error="missing")],
        ],
        require_complete=True,
    )
    assert [row["item_id"] for row in merged] == ["a", "b", "c"]
    assert merged[1]["low_level"]["motion_mean"] == 2
    assert summary == {
        "manifest_items": 3,
        "input_attempt_records": 4,
        "duplicate_attempt_records": 1,
        "canonical_items": 3,
        "successful_items": 2,
        "failed_or_missing_media_items": 1,
        "unattempted_items": 0,
        "complete": True,
        "corpus_mutated": False,
    }


def test_merge_fails_closed_for_incomplete_or_unknown_items():
    manifest = [{"item_id": "a"}, {"item_id": "b"}]
    with pytest.raises(ValueError, match="missing score attempts"):
        merge_scores(manifest, [[score("a", 1)]], require_complete=True)
    with pytest.raises(ValueError, match="unknown item"):
        merge_scores(
            manifest,
            [[score("a", 1), score("b", 1), score("outside", 1)]],
            require_complete=True,
        )


def test_merge_allows_partial_ledger_when_requested():
    merged, summary = merge_scores(
        [{"item_id": "a"}, {"item_id": "b"}],
        [[score("b", 1)]],
        require_complete=False,
    )
    assert [row["item_id"] for row in merged] == ["b"]
    assert summary["unattempted_items"] == 1
    assert summary["complete"] is False
