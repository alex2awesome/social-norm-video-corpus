from scripts.evaluate_rank_confirmation_semantic_bands import evaluate


def test_instructional_summarizes_strict_and_visual_by_band():
    sealed = [
        {"item_id": "a", "uid": "u1", "score_band": "top_quintile"},
        {"item_id": "b", "uid": "u2", "score_band": "bottom_quintile"},
    ]
    reviews = [
        {
            "item_id": "a",
            "strict_instructional_pass": "yes",
            "visual_demo_present": "yes",
            "assigned_norm_is_social_norm": "yes",
            "recoverable_route": "instructional",
        },
        {
            "item_id": "b",
            "strict_instructional_pass": "no",
            "visual_demo_present": "no",
            "assigned_norm_is_social_norm": "yes",
            "recoverable_route": "commentary",
        },
    ]
    report = evaluate(sealed, reviews, "instructional")
    assert report["strict"]["rate"] == 0.5
    assert report["bands"]["top_quintile"]["visual_demo"]["rate"] == 1
    assert report["bands"]["bottom_quintile"]["routes"] == {"commentary": 1}


def test_witnessed_evaluates_top_band_after_title_staging_exclusion():
    sealed = [
        {"item_id": "a", "uid": "u1", "score_band": "top_quintile"},
        {"item_id": "b", "uid": "u2", "score_band": "top_quintile"},
        {"item_id": "c", "uid": "u3", "score_band": "bottom_quintile"},
    ]
    reviews = [
        {
            "item_id": item,
            "strict_witnessed_pass": "no",
            "recoverable_route": "instructional",
        }
        for item in ("a", "b", "c")
    ]
    blind = [
        {"item_id": "a", "witnessed_candidate_blind": "no"},
        {"item_id": "b", "witnessed_candidate_blind": "yes"},
        {"item_id": "c", "witnessed_candidate_blind": "no"},
    ]
    staging = [
        {"uid": "u1", "title_staging_cue": True},
        {"uid": "u2", "title_staging_cue": False},
        {"uid": "u3", "title_staging_cue": False},
    ]
    report = evaluate(
        sealed,
        reviews,
        "witnessed",
        blind=blind,
        staging=staging,
        production_adjudications=[
            {
                "item_id": "b",
                "witnessed_candidate_post_reveal": "no",
            }
        ],
    )
    combined = report["selections"][
        "top_quintile_without_title_staging_cue"
    ]
    assert combined["selected"] == 1
    assert combined["organic_candidate"]["precision"] == 0
    assert combined["organic_candidate"]["recall"] is None
    assert report["selections"]["post_reveal_production_adjudications"] == 1
