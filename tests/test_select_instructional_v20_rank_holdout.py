from scripts.select_instructional_v20_rank_holdout import blind_record, select


def test_select_freezes_disjoint_requested_bands() -> None:
    rows = [
        {
            "item_id": f"i{index}",
            "visual_rank_score": index / 100,
            "development_review_band": index >= 60,
        }
        for index in range(100)
    ]
    selected = select(rows, 10, 5, 5, "seed")
    assert len(selected) == 20
    assert len({row["item_id"] for row in selected}) == 20
    bands = [row["sealed_rank_band"] for row in selected]
    assert bands.count("high") == 10
    assert bands.count("boundary_below") == 5
    assert bands.count("low") == 5
    boundary_scores = sorted(
        row["visual_rank_score"]
        for row in selected
        if row["sealed_rank_band"] == "boundary_below"
    )
    assert boundary_scores == [0.55, 0.56, 0.57, 0.58, 0.59]


def test_select_excludes_prior_uids_and_uses_each_source_once() -> None:
    rows = [
        {
            "item_id": f"i{index}",
            "uid": "excluded" if index == 99 else f"u{index // 2}",
            "visual_rank_score": index / 100,
            "development_review_band": index >= 60,
        }
        for index in range(100)
    ]
    selected = select(rows, 5, 5, 5, "seed", {"excluded"})
    assert len(selected) == 15
    assert len({row["uid"] for row in selected}) == 15
    assert all(row["uid"] != "excluded" for row in selected)


def test_blind_record_accepts_canonical_manifest_without_per_clip_hash() -> None:
    record = blind_record(
        {
            "item_id": "instructional:u:0",
            "uid": "u",
            "pillar": "instructional",
            "source_clip": "/clips/u.mp4",
        },
        3,
        "audit-0003",
    )
    assert record["source_clip"] == "/clips/u.mp4"
    assert "source_sha256" not in record
