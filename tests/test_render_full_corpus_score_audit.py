from scripts.render_full_corpus_score_audit import (
    select_score_bands,
    select_uniform_sample,
)


def test_score_band_selection_is_balanced_deterministic_and_source_disjoint():
    manifest = []
    scores = []
    for index in range(60):
        item_id = f"item:{index}"
        manifest.append(
            {
                "item_id": item_id,
                "uid": f"uid:{index}",
                "source_exists": True,
            }
        )
        value = index / 59
        scores.append(
            {
                "item_id": item_id,
                "error": None,
                "low_level": {
                    "motion_mean": value,
                    "histogram_delta_mean": value,
                },
            }
        )
    first = select_score_bands(manifest, scores, 5, "seed")
    second = select_score_bands(manifest, scores, 5, "seed")
    assert [row["item_id"] for row in first] == [
        row["item_id"] for row in second
    ]
    assert len(first) == 15
    assert len({row["uid"] for row in first}) == 15
    counts = {}
    for row in first:
        counts[row["_score_band"]] = counts.get(row["_score_band"], 0) + 1
    assert counts == {
        "low_activity": 5,
        "middle_activity": 5,
        "high_activity": 5,
    }


def test_score_band_selection_excludes_prior_uids_and_round_robins_strata():
    manifest = []
    scores = []
    for index in range(120):
        item_id = f"item:{index}"
        manifest.append(
            {
                "item_id": item_id,
                "uid": f"uid:{index}",
                "source_exists": True,
                "polarity": ("violation", "correct", "explanation")[index % 3],
            }
        )
        value = index / 119
        scores.append(
            {
                "item_id": item_id,
                "error": None,
                "low_level": {
                    "motion_mean": value,
                    "histogram_delta_mean": value,
                },
            }
        )
    selected = select_score_bands(
        manifest,
        scores,
        6,
        "stratified",
        excluded_uids={"uid:0", "uid:60", "uid:119"},
        stratify_fields=("polarity",),
    )
    assert not ({row["uid"] for row in selected} & {"uid:0", "uid:60", "uid:119"})
    for band in ("low_activity", "middle_activity", "high_activity"):
        values = {
            row["polarity"]
            for row in selected
            if row["_score_band"] == band
        }
        assert values == {"violation", "correct", "explanation"}
        assert all(row["_stratum"] == {"polarity": row["polarity"]} for row in selected if row["_score_band"] == band)


def test_uniform_selection_is_deterministic_source_disjoint_and_covers_quintiles():
    manifest = []
    scores = []
    for index in range(100):
        # Two items per source exercises source-disjoint selection.
        item_id = f"item:{index}"
        manifest.append(
            {
                "item_id": item_id,
                "uid": f"uid:{index // 2}",
                "source_exists": True,
            }
        )
        value = index / 99
        scores.append(
            {
                "item_id": item_id,
                "error": None,
                "low_level": {
                    "motion_mean": value,
                    "histogram_delta_mean": value,
                },
            }
        )
    first = select_uniform_sample(
        manifest, scores, 30, "uniform", excluded_uids={"uid:0"}
    )
    second = select_uniform_sample(
        manifest, scores, 30, "uniform", excluded_uids={"uid:0"}
    )
    assert [row["item_id"] for row in first] == [
        row["item_id"] for row in second
    ]
    assert len({row["uid"] for row in first}) == 30
    assert "uid:0" not in {row["uid"] for row in first}
    assert all(row["_score_band"].endswith("quintile") for row in first)
