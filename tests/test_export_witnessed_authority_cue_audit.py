from __future__ import annotations

from scripts.export_witnessed_authority_cue_audit import (
    balanced_source_disjoint,
    cue_stratum,
)


def row(uid, cue, cue_ids=(), role="camera_person"):
    return {
        "uid": uid,
        "item_id": f"witnessed:{uid}:clip_0",
        "authority_reaction_cue": cue,
        "authority_cue_ids": list(cue_ids),
        "reactor_role_provenance": role,
        "reaction_observations": [{}],
    }


def test_cue_strata_prefer_explicit_phrase_then_tag():
    assert cue_stratum(row("a", True, ["phrase:under_arrest"])) == "phrase"
    assert cue_stratum(row("b", True, ["tag:command"])) == "tag"
    assert cue_stratum(row("c", True, ["norm:authority"])) == "norm"


def test_selection_is_balanced_and_source_disjoint():
    rows = [
        row("p1", True, ["phrase:under_arrest"]),
        row("p2", True, ["tag:command"]),
        row("p3", True, ["norm:authority"]),
        row("p4", True, ["phrase:hands_command"]),
        row("n1", False, role="camera_person"),
        row("n2", False, role="mixed"),
        row("n3", False, role="bystander"),
        row("n4", False, role="victim"),
    ]
    selected = balanced_source_disjoint(rows, 3, 3, "seed")
    assert sum(item["authority_reaction_cue"] for item in selected) == 3
    assert sum(not item["authority_reaction_cue"] for item in selected) == 3
    assert len({item["uid"] for item in selected}) == 6


def test_selection_excludes_prior_sources():
    rows = [
        row("old_positive", True, ["phrase:under_arrest"]),
        row("new_positive", True, ["phrase:under_arrest"]),
        row("old_negative", False),
        row("new_negative", False),
    ]
    selected = balanced_source_disjoint(
        rows,
        1,
        1,
        "seed",
        excluded_uids={"old_positive", "old_negative"},
    )
    assert {item["uid"] for item in selected} == {
        "new_positive",
        "new_negative",
    }
