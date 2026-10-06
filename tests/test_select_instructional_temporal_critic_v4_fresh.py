from scripts.select_instructional_temporal_critic_v4_fresh import select


def row(index, uid=None, category="instr_family", polarity="violation"):
    uid = uid or f"dailymotion__{index}"
    return {
        "item_id": f"instructional:{uid}:{index}", "uid": uid,
        "source_clip": f"/clips/{uid}.mp4", "source_platform": uid.split("__")[0],
        "category": category, "polarity": polarity,
    }


def test_selection_is_fresh_source_disjoint_and_deterministic():
    rows = [row(i, category=f"instr_{i % 3}", polarity=("correct" if i % 2 else "violation")) for i in range(20)]
    first = select(rows, {"dailymotion__0", "dailymotion__1"}, 10, "seed")
    second = select(list(reversed(rows)), {"dailymotion__0", "dailymotion__1"}, 10, "seed")
    assert [x["item_id"] for x in first] == [x["item_id"] for x in second]
    assert len({x["uid"] for x in first}) == 10
    assert not {x["uid"] for x in first} & {"dailymotion__0", "dailymotion__1"}


def test_one_demo_per_source_prevents_source_leakage():
    rows = [row(0, "dailymotion__same"), row(1, "dailymotion__same"), row(2)]
    selected = select(rows, set(), 2, "seed")
    assert len({x["uid"] for x in selected}) == 2
