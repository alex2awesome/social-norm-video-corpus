from scripts.cross_pillar_shadow_supervision_v1 import score_record


def _all_yes(pillar: str) -> dict:
    from scripts.cross_pillar_shadow_supervision_v1 import FAMILY_FIELDS
    return {"item_id": "x", "pillar": pillar, **{
        field: "yes" for fields in FAMILY_FIELDS[pillar].values() for field in fields
    }}


def test_complete_candidate_is_still_not_automatically_accepted() -> None:
    result = score_record(_all_yes("witnessed"))
    assert result["review_band"] == "complete_manual_review_candidate"
    assert result["automatic_acceptance"] is False
    assert result["corpus_disposition"] is None
    assert result["delete_media"] is False


def test_correlated_atomic_observations_collapse_to_one_family() -> None:
    result = score_record(_all_yes("instructional"))
    assert len(result["positive_families"]) == 6
    assert "social_scope" in result["lf_families"]
    assert len(result["lf_families"]["social_scope"]["fields"]) == 4


def test_negative_witnessed_role_signal_forces_reroute_review() -> None:
    row = _all_yes("witnessed")
    row["distinct_bystander"] = "no"
    result = score_record(row)
    assert result["lf_families"]["distinct_reactor"]["vote"] == "negative"
    assert result["review_band"] == "likely_reroute_or_reject_review"


def test_uncertain_fields_abstain_instead_of_becoming_negative() -> None:
    row = _all_yes("commentary")
    row["event_visible"] = "uncertain"
    result = score_record(row)
    assert result["lf_families"]["visual_localization"]["vote"] == "abstain"
    assert not result["negative_families"]

