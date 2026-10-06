from scripts.materialize_witnessed_corpus_review_routes import materialize


def test_routes_audited_cues_without_accepting_uncued_rows() -> None:
    authority = [
        {
            "item_id": "witnessed:a:clip_0",
            "uid": "a",
            "clip_idx": 0,
            "authority_reaction_cue": False,
        },
        {
            "item_id": "witnessed:b:clip_0",
            "uid": "b",
            "clip_idx": 0,
            "authority_reaction_cue": True,
            "authority_cue_ids": ["phrase:hands_command"],
        },
    ]
    staging = [
        {
            "uid": "a",
            "strict_organic_witnessed_eligible": False,
            "source_staging_title_cue": True,
        },
        {"uid": "b", "strict_organic_witnessed_eligible": None},
    ]
    rows, summary = materialize(authority, staging)
    assert rows[0]["review_routes"] == ["instructional_demo_review"]
    assert rows[1]["review_routes"] == ["visible_scene_review"]
    assert all(row["strict_organic_witnessed_eligible"] is False for row in rows)
    assert summary["strict_excluded_by_audited_cue"] == 2


def test_overlapping_cues_preserve_both_review_routes() -> None:
    authority = [
        {
            "item_id": "witnessed:a:clip_0",
            "uid": "a",
            "clip_idx": 0,
            "authority_reaction_cue": True,
        }
    ]
    staging = [
        {"uid": "a", "strict_organic_witnessed_eligible": False}
    ]
    rows, summary = materialize(authority, staging)
    assert rows[0]["review_routes"] == [
        "instructional_demo_review",
        "visible_scene_review",
    ]
    assert summary["overlapping_reroute_cues"] == 1


def test_uncued_row_remains_unresolved_not_accepted() -> None:
    rows, summary = materialize(
        [
            {
                "item_id": "witnessed:a:clip_0",
                "uid": "a",
                "authority_reaction_cue": False,
            }
        ],
        [{"uid": "a", "strict_organic_witnessed_eligible": None}],
    )
    assert rows[0]["strict_organic_witnessed_eligible"] is None
    assert rows[0]["review_routes"] == ["strict_witnessed_unresolved_review"]
    assert summary["strict_unresolved"] == 1
