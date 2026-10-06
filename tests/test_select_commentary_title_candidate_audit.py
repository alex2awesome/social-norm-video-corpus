import pytest

from scripts.select_commentary_title_candidate_audit import select


def row(index: int, cues: list[str], uid: str | None = None) -> dict:
    return {
        "item_id": f"commentary:u{index}:0",
        "uid": uid or f"u{index}",
        "title_event_cues": cues,
    }


def test_selection_is_source_disjoint_stratified_and_excludes_prior_uids() -> None:
    rows = [
        row(index, ["actor_action"] if index % 2 else ["capture_before_event"])
        for index in range(20)
    ]
    rows.append(row(99, ["actor_action"], uid="excluded"))
    selected = select(rows, 10, "seed", {"excluded"})
    assert len(selected) == 10
    assert len({item["uid"] for item in selected}) == 10
    assert {tuple(item["title_event_cues"]) for item in selected} == {
        ("actor_action",),
        ("capture_before_event",),
    }
    assert [item["audit_index"] for item in selected] == list(range(10))


def test_selection_fails_if_unique_source_coverage_is_insufficient() -> None:
    with pytest.raises(ValueError, match="wanted 2"):
        select([row(0, ["actor_action"], uid="same"), row(1, ["actor_action"], uid="same")], 2, "seed")
