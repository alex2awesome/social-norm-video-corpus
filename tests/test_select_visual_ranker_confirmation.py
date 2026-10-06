from scripts.select_visual_ranker_confirmation import choose_score_bands


def test_choose_score_bands_is_source_disjoint_and_balanced():
    rows = [
        {
            "item_id": f"p:u{index}:{index}",
            "uid": f"u{index}",
            "visual_score": index / 100,
        }
        for index in range(100)
    ]
    selected = choose_score_bands(rows, per_band=5, seed="test")
    assert len(selected) == 15
    assert len({row["uid"] for row in selected}) == 15
    assert {
        row["score_band"] for row in selected
    } == {"bottom_quintile", "middle_quintile", "top_quintile"}
