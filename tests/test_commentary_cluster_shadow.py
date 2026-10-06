from scripts.run_commentary_cluster_shadow import source_uid, validate_clusters


def rows():
    return [
        {"item_id": "commentary:dailymotion__abc:0"},
        {"item_id": "commentary:dailymotion__abc:1"},
    ]


def test_source_uid_preserves_platform_uid():
    assert source_uid("commentary:dailymotion__abc:7") == "dailymotion__abc"


def test_cluster_contract_requires_exact_partition():
    valid, errors = validate_clusters(rows(), [{
        "member_item_ids": ["commentary:dailymotion__abc:0", "commentary:dailymotion__abc:1"],
        "representative_item_id": "commentary:dailymotion__abc:0",
    }])
    assert valid is True
    assert errors == []


def test_cluster_contract_rejects_missing_duplicate_and_cross_video():
    input_rows = rows() + [{"item_id": "commentary:dailymotion__xyz:0"}]
    valid, errors = validate_clusters(input_rows, [{
        "member_item_ids": ["commentary:dailymotion__abc:0", "commentary:dailymotion__abc:0", "commentary:dailymotion__xyz:0"],
        "representative_item_id": "commentary:dailymotion__abc:0",
    }])
    assert valid is False
    assert any("crosses_source_videos" in error for error in errors)
    assert any("duplicate_members" in error for error in errors)
    assert any("missing_members" in error for error in errors)
