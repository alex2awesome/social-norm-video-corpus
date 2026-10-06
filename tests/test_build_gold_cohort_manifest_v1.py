import json

import pytest

from scripts.build_gold_cohort_manifest_v1 import (
    blind_ledger_row,
    build_outputs,
    parse_stratum_args,
    reveal_ledger_row,
    select_cohort,
)


def inventory(n=40):
    rows = []
    for i in range(n):
        pillar = ("witnessed", "instructional", "commentary")[i % 3]
        rows.append(
            {
                "uid": f"yt__{i}",
                "pillar": pillar,
                "source_uid": f"src{i // 2}",  # two items share a source
                "channel": f"chan{i % 5}",
                "cluster": f"cluster{i % 8}",
                "strata": ["high_score"] if i % 4 == 0 else ["low_score"],
                "media_ref": f"data/hits/yt__{i}/clip_0.mp4",
                "transcript": "SHOULD NOT LEAK INTO BLIND LEDGER",
                "assigned_norm": "queue jumping",
            }
        )
    return rows


def test_selection_is_deterministic_and_source_disjoint():
    rows = inventory()
    a, summary = select_cohort(
        rows, seed="wave1", stratum_counts={"uniform": 10}, excluded_uids=set()
    )
    b, _ = select_cohort(
        rows, seed="wave1", stratum_counts={"uniform": 10}, excluded_uids=set()
    )
    assert [r["uid"] for r in a] == [r["uid"] for r in b]
    sources = [r["source_uid"] for r in a]
    clusters = [r["cluster"] for r in a]
    assert len(set(sources)) == len(sources)
    assert len(set(clusters)) == len(clusters)
    assert summary["selected_total"] == len(a)
    # A different seed produces a different ordering.
    c, _ = select_cohort(
        rows, seed="wave2", stratum_counts={"uniform": 10}, excluded_uids=set()
    )
    assert [r["uid"] for r in a] != [r["uid"] for r in c]


def test_channel_cap_exclusions_and_strata():
    rows = inventory()
    selected, summary = select_cohort(
        rows,
        seed="wave1",
        stratum_counts={"high_score": 5, "uniform": 8},
        excluded_uids={"yt__0", "yt__4"},
        channel_cap=1,
    )
    uids = {r["uid"] for r in selected}
    assert not uids & {"yt__0", "yt__4"}
    channels = [r["channel"] for r in selected]
    assert all(channels.count(channel) <= 1 for channel in channels)
    high = [r for r in selected if r["stratum"] == "high_score"]
    assert all("high_score" in r["strata"] for r in high)
    # With a channel cap of 1 and 5 channels, at most 5 items total fit.
    assert summary["selected_total"] <= 5
    assert summary["shortfalls"]


def test_stratum_fill_shortfall_is_reported_not_padded():
    rows = inventory(6)
    _, summary = select_cohort(
        rows, seed="s", stratum_counts={"disagreement": 3}, excluded_uids=set()
    )
    assert summary["selected_total"] == 0
    assert summary["shortfalls"] == {"disagreement": 3}


def test_unknown_stratum_and_bad_inventory_rejected():
    with pytest.raises(ValueError, match="unknown strata"):
        select_cohort(
            inventory(3), seed="s", stratum_counts={"vibes": 1}, excluded_uids=set()
        )
    with pytest.raises(ValueError, match="missing source_uid"):
        select_cohort(
            [{"uid": "a", "pillar": "witnessed"}],
            seed="s",
            stratum_counts={"uniform": 1},
            excluded_uids=set(),
        )
    with pytest.raises(ValueError, match="invalid pillar"):
        select_cohort(
            [{"uid": "a", "pillar": "negatives", "source_uid": "s"}],
            seed="s",
            stratum_counts={"uniform": 1},
            excluded_uids=set(),
        )


def test_blind_ledger_carries_no_semantic_leakage():
    rows = inventory(3)
    selected, _ = select_cohort(
        rows, seed="s", stratum_counts={"uniform": 3}, excluded_uids=set()
    )
    for row in selected:
        blind = blind_ledger_row(row)
        assert "transcript" not in blind
        assert "assigned_norm" not in blind
        assert "strata" not in blind and "stratum" not in blind
        assert blind["grounding_mode"] == ""
        assert blind["blind_review_complete"] is False
        if row["pillar"] == "witnessed":
            assert "responder_role" in blind
        reveal = reveal_ledger_row(row)
        assert reveal["label_relation"] == ""
        assert reveal["blind_row_frozen"] is False


def test_outputs_are_append_only_and_hash_sealed(tmp_path):
    selected, summary = select_cohort(
        inventory(9), seed="s", stratum_counts={"uniform": 4}, excluded_uids=set()
    )
    out = tmp_path / "wave"
    final = build_outputs(selected, summary, out)
    assert len(final["manifest_sha256"]) == 64
    manifest_rows = [
        json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(manifest_rows) == 4
    with pytest.raises(FileExistsError):
        build_outputs(selected, summary, out)


def test_parse_stratum_args():
    assert parse_stratum_args(["uniform=10", "high_score=5"]) == {
        "uniform": 10,
        "high_score": 5,
    }
    with pytest.raises(ValueError, match="invalid stratum"):
        parse_stratum_args(["uniform=zero"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_stratum_args(["uniform=1", "uniform=2"])
    with pytest.raises(ValueError, match="at least one"):
        parse_stratum_args([])
