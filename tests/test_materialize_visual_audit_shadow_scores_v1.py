from scripts.materialize_visual_audit_shadow_scores_v1 import atomic_from_review


def test_sparse_positive_abstains_on_unreviewed_families() -> None:
    row = atomic_from_review({"pillar": "instructional", "uid": "u",
                              "visual_target": "yes", "failure": "none"})
    assert row["lf_families"]["demonstration"]["vote"] == "abstain"
    assert row["automatic_acceptance"] is False


def test_authority_failure_is_a_named_negative_family() -> None:
    row = atomic_from_review({"pillar": "witnessed", "uid": "u",
                              "visual_target": "no", "failure": "authority_not_bystander"})
    assert row["lf_families"]["distinct_reactor"]["vote"] == "negative"
    assert row["review_band"] == "likely_reroute_or_reject_review"


def test_visible_commentary_does_not_claim_temporal_localization() -> None:
    row = atomic_from_review({"pillar": "commentary", "uid": "u",
                              "visual_target": "yes", "failure": "none"})
    assert row["event_visible"] == "yes"
    assert row["event_temporally_localized"] == "uncertain"
    assert row["lf_families"]["visual_localization"]["vote"] == "abstain"
