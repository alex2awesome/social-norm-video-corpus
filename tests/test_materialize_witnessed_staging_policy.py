from scripts.materialize_witnessed_staging_policy import materialize


def test_positive_cue_excludes_only_strict_organic_route():
    row = materialize(
        {
            "uid": "u",
            "title": "Phone prank",
            "title_staging_cue": True,
            "matches": {"title": ["prank"]},
            "error": None,
        }
    )
    assert row["strict_organic_witnessed_eligible"] is False
    assert row["next_review"] == "instructional_demo_gates"


def test_negative_cue_is_undecided_not_accepted():
    row = materialize(
        {
            "uid": "u",
            "title": "Incident",
            "title_staging_cue": False,
            "matches": {"title": []},
            "error": None,
        }
    )
    assert row["strict_organic_witnessed_eligible"] is None


def test_audited_creator_initiated_cue_excludes_strict_organic_route():
    row = materialize(
        {
            "uid": "u",
            "title": "Trying to kiss strangers",
            "title_staging_cue": False,
            "title_creator_initiated_candidate_cue": True,
            "matches": {
                "title": [],
                "title_creator_initiated_candidate": ["Trying to kiss strangers"],
            },
            "error": None,
        }
    )
    assert row["source_staging_title_cue"] is False
    assert row["source_creator_initiated_title_cue"] is True
    assert row["strict_organic_witnessed_eligible"] is False
    assert row["next_review"] == "instructional_demo_gates"
