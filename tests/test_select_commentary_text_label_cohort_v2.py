from scripts.select_commentary_text_label_cohort_v2 import select


def row(index, platform="dailymotion", category="family"):
    uid = f"{platform}__{index}"
    return {
        "uid": uid, "item_id": f"commentary:{uid}:0",
        "source_platform": platform, "query_source": f"q{index % 2}",
        "category": category, "signal": f"s{index % 3}",
    }


def test_selection_round_robins_platforms_and_is_deterministic():
    rows = [row(i) for i in range(10)] + [row(i, "youtube") for i in range(10, 20)]
    first = select(rows, 8, "seed")
    second = select(list(reversed(rows)), 8, "seed")
    assert [x["item_id"] for x in first] == [x["item_id"] for x in second]
    assert sum(x["source_platform"] == "dailymotion" for x in first) == 4
    assert sum(x["source_platform"] == "youtube" for x in first) == 4
    assert len({x["uid"] for x in first}) == 8


def test_selection_uses_available_platform_when_another_exhausts():
    rows = [row(0, "youtube")] + [row(i, "dailymotion") for i in range(1, 8)]
    selected = select(rows, 6, "seed")
    assert len(selected) == 6
    assert sum(x["source_platform"] == "youtube" for x in selected) == 1
