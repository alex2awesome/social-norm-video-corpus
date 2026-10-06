import pytest

from scripts.select_instructional_visual_title_transfer_v1 import select


def row(index, title, category="a", polarity="correct", uid=None):
    uid = uid or f"u{index}"
    return {
        "item_id": f"instructional:{uid}:{index}", "uid": uid,
        "source_clip": f"/{index}.mp4", "title": title,
        "category": category, "polarity": polarity,
    }


def test_selection_is_deterministic_stratified_and_source_disjoint() -> None:
    rows = []
    for index in range(12):
        rows.append(row(
            index, "Roleplay scenario" if index < 6 else "General lecture",
            category=("a", "b")[index % 2],
            polarity=("correct", "violation")[index % 2],
        ))
    first = select(rows, 4, 3, "seed", {"u0"})
    second = select(rows, 4, 3, "seed", {"u0"})
    assert [r["item_id"] for r in first] == [r["item_id"] for r in second]
    assert len(first) == len({r["uid"] for r in first}) == 7
    assert sum(r["scene_title_candidate"] for r in first) == 4
    assert "u0" not in {r["uid"] for r in first}


def test_only_one_demo_per_source_can_be_selected() -> None:
    rows = [
        row(0, "Roleplay scenario", uid="shared"),
        row(1, "Roleplay scenario", uid="shared"),
        row(2, "General lecture"),
    ]
    chosen = select(rows, 1, 1, "seed", set())
    assert len({r["uid"] for r in chosen}) == 2


def test_insufficient_title_stratum_fails_closed() -> None:
    with pytest.raises(ValueError, match="insufficient rows"):
        select([row(0, "General lecture")], 1, 0, "seed", set())
