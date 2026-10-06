from scripts.render_strict_audit_calibration_v1 import activity_band, select


def test_activity_bands_leave_buffers_between_cells() -> None:
    assert activity_band(0.25) == "low"
    assert activity_band(0.30) is None
    assert activity_band(0.50) == "middle"
    assert activity_band(0.70) is None
    assert activity_band(0.75) == "high"


def test_select_is_source_disjoint_and_balanced() -> None:
    rows = []
    for pillar in ("instructional", "witnessed", "commentary"):
        for origin in ("baseline", "delta"):
            for band in ("low", "middle", "high"):
                for index in range(3):
                    rows.append(
                        {
                            "pillar": pillar,
                            "_origin": origin,
                            "_activity_band": band,
                            "uid": f"{pillar}-{origin}-{band}-{index}",
                        }
                    )
    chosen = select(rows, 2, "seed")
    assert len(chosen) == 36
    assert len({row["uid"] for row in chosen}) == 36


def test_select_excludes_prior_sources() -> None:
    rows = []
    excluded = set()
    for pillar in ("instructional", "witnessed", "commentary"):
        for origin in ("baseline", "delta"):
            for band in ("low", "middle", "high"):
                old = f"old-{pillar}-{origin}-{band}"
                excluded.add(old)
                for uid in (old, f"new-{pillar}-{origin}-{band}"):
                    rows.append(
                        {
                            "pillar": pillar,
                            "_origin": origin,
                            "_activity_band": band,
                            "uid": uid,
                        }
                    )
    chosen = select(rows, 1, "seed", excluded)
    assert len(chosen) == 18
    assert not ({row["uid"] for row in chosen} & excluded)
