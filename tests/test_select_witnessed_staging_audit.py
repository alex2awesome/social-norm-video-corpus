from pathlib import Path

from scripts.select_witnessed_staging_audit import cue_group, select


def score(uid: str, *, title=False, reveal=False, creator=False):
    return {
        "uid": uid,
        "explicit_staging_candidate": title or reveal or creator,
        "title_staging_cue": title,
        "transcript_explicit_reveal_cue": reveal,
        "transcript_creator_setup_cue": creator,
        "error": None,
        "matches": {},
    }


def add_clip(root: Path, uid: str):
    path = root / "data" / "hits" / uid / "clip_0.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")


def test_cue_group_is_stable():
    assert cue_group(score("u", title=True, creator=True)) == "title+creator"


def test_select_balances_groups_and_excludes_sources(tmp_path: Path):
    for uid in ("a", "b", "c", "d"):
        add_clip(tmp_path, uid)
    rows = [
        score("a", title=True),
        score("b", title=True),
        score("c", reveal=True),
        score("d", reveal=True),
    ]
    result = select(rows, tmp_path, {"b"}, 1, "seed")
    assert len(result) == 2
    assert {row["uid"] for row in result} & {"a"} == {"a"}
    assert len({row["uid"] for row in result} & {"c", "d"}) == 1
    assert all("clip" in row for row in result)


def test_select_can_target_candidate_only_field(tmp_path: Path):
    add_clip(tmp_path, "candidate")
    row = score("candidate")
    row["title_creator_initiated_candidate_cue"] = True
    result = select(
        [row],
        tmp_path,
        set(),
        10,
        "seed",
        "title_creator_initiated_candidate_cue",
    )
    assert [selected["uid"] for selected in result] == ["candidate"]
    assert result[0]["cue_group"] == "creator_initiated_title"


def test_select_can_target_wwyd_candidate_field(tmp_path: Path):
    add_clip(tmp_path, "wwyd")
    row = score("wwyd")
    row["title_wwyd_candidate_cue"] = True
    result = select(
        [row], tmp_path, set(), 10, "seed", "title_wwyd_candidate_cue"
    )
    assert [selected["uid"] for selected in result] == ["wwyd"]
    assert result[0]["cue_group"] == "wwyd_title"
