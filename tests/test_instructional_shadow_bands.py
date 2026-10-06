from scripts.assign_instructional_shadow_bands import assign_band


def visual(scene="yes", domain="yes", label="yes", localization="clean"):
    return {
        "result": {
            "usable_demo_after_relabel": scene,
            "social_norm_domain": domain,
            "proposed_norm_supported": label,
            "localization_quality": localization,
        }
    }


def text(social="yes"):
    return {"result": {"social_norm_candidate": social}}


def test_band_requires_complete_independent_coverage():
    assert assign_band(visual(), None, text()) == "incomplete_score_coverage"


def test_high_confidence_band_requires_dual_clean_label_and_text_agreement():
    assert assign_band(visual(), visual(), text()) == "dual_clean_label_candidate"
    assert (
        assign_band(visual(), visual(), text(), duration_hint=181)
        == "dual_clean_long_context_review"
    )
    assert (
        assign_band(visual(label="no"), visual(), text())
        == "dual_clean_relabel_review"
    )
    assert (
        assign_band(visual(), visual(), text("no"))
        == "dual_clean_text_conflict"
    )


def test_broad_and_disagreement_are_review_queues():
    assert (
        assign_band(visual(localization="broad"), visual(), text())
        == "dual_scene_recut_review"
    )
    assert (
        assign_band(visual(), visual(scene="no"), text())
        == "single_vlm_candidate_review"
    )
    assert (
        assign_band(visual(scene="no"), visual(scene="no"), text())
        == "text_positive_visual_rescue_review"
    )
    assert (
        assign_band(visual(scene="no"), visual(scene="no"), text("no"))
        == "low_evidence_manual_sample"
    )
