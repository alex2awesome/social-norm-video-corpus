from scripts.render_commentary_title_population_dense import select_remaining


def test_select_remaining_is_exhaustive_disjoint_and_deterministic() -> None:
    candidates = [
        {"item_id": f"i{i}", "uid": f"u{i}", "norm": f"title {i}"}
        for i in range(5)
    ]
    audited = [{"item_id": "i1"}, {"item_id": "i3"}]

    first = select_remaining(candidates, audited, "seed")
    second = select_remaining(candidates, audited, "seed")

    assert first == second
    assert {row["item_id"] for row in first} == {"i0", "i2", "i4"}
    assert [row["audit_index"] for row in first] == [0, 1, 2]
    assert len({row["candidate_id"] for row in first}) == 3
