from scripts.materialize_successful_shadow_jsonl import canonicalize


def test_last_success_wins_and_failures_remain_explicit():
    manifest = [{"item_id": "a"}, {"item_id": "b"}, {"item_id": "c"}]
    attempts = [
        {"item_id": "a", "result": None, "error": "bad"},
        {"item_id": "a", "result": {"value": 1}, "error": None},
        {"item_id": "a", "result": {"value": 2}, "error": None},
        {"item_id": "b", "result": None, "error": "bad"},
        {"item_id": "outside", "result": {"value": 9}, "error": None},
    ]

    rows, summary = canonicalize(manifest, attempts)

    assert rows == [{"item_id": "a", "result": {"value": 2}, "error": None}]
    assert summary["canonical_successes"] == 1
    assert summary["missing_or_error_only"] == ["b", "c"]
    assert summary["error_only_items"] == ["b"]
    assert summary["unexpected_items"] == ["outside"]
    assert summary["duplicate_attempt_rows"] == 2
    assert summary["items_with_multiple_successes"] == ["a"]


def test_duplicate_manifest_ids_are_rejected():
    try:
        canonicalize([{"item_id": "a"}, {"item_id": "a"}], [])
    except ValueError as exc:
        assert "duplicate item_id" in str(exc)
    else:
        raise AssertionError("duplicate manifest IDs were accepted")
