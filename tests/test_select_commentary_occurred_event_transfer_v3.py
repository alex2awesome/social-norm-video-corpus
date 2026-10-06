import pytest

from scripts.select_commentary_occurred_event_transfer_v3 import select


def row(uid, decision, platform="youtube", query_source="seed", item_suffix="0"):
    return {
        "uid": uid,
        "item_id": f"commentary:{uid}:{item_suffix}",
        "v1_decision": decision,
        "source_platform": platform,
        "query_source": query_source,
        "source_manifest_path": f"runs/{uid}/manifest.json",
    }


def test_balances_decisions_excludes_design_sources_and_is_source_disjoint():
    rows = [
        row("a", "accept"),
        row("b", "accept_after_relabel", "dailymotion"),
        row("c", "reject", "reddit"),
        row("d", "reject", "rumble"),
        row("excluded", "accept"),
        row("a", "reject", item_suffix="1"),
    ]
    selected = select(rows, {"excluded"}, 2, 2, "seed")
    assert len(selected) == 4
    assert len({item["uid"] for item in selected}) == 4
    assert sum(item["v1_decision"] in {"accept", "accept_after_relabel"} for item in selected) == 2
    assert sum(item["v1_decision"] == "reject" for item in selected) == 2
    assert all(item["manual_reaudit_required"] is True for item in selected)


def test_fails_closed_when_requested_stratum_is_too_small():
    with pytest.raises(ValueError, match="insufficient"):
        select([row("a", "accept")], set(), 1, 1, "seed")
