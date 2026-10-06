import pytest

from scripts.evaluate_instructional_av_ensemble import (
    evaluate,
    primary_with_fallbacks,
    unique_by_item,
)


def manual(item_id: str, *, visual: str, social: str, label: str) -> dict:
    return {
        "item_id": item_id,
        "visual_demo_present": visual,
        "is_social_norm": social,
        "norm_supported": label,
        "decision": "accept" if visual == "yes" else "reject_described_only",
        "evidence": "manual evidence",
    }


def prediction(item_id: str, result: dict, *, error=None) -> dict:
    return {"item_id": item_id, "result": result, "error": error}


def test_unique_by_item_retries_failed_append_only_attempt():
    failed = prediction("x", {}, error="timeout")
    failed["result"] = None
    success = prediction("x", {"ok": True})

    assert unique_by_item([failed, success], source="test") == {"x": success}


def test_fallback_fills_missing_but_does_not_override_primary():
    primary = prediction("x", {"source": "primary"})
    fallback_x = prediction("x", {"source": "fallback"})
    fallback_y = prediction("y", {"source": "fallback"})

    rows = primary_with_fallbacks(
        [primary],
        [[fallback_x, fallback_y]],
        source="test",
    )

    assert {row["item_id"]: row["result"]["source"] for row in rows} == {
        "x": "primary",
        "y": "fallback",
    }


def test_evaluator_refuses_partial_predictions():
    m = [manual("x", visual="yes", social="yes", label="yes")]
    video = prediction(
        "x",
        {
            "situated_social_scenario_visible": "yes",
            "proposed_norm_plausibly_demonstrated": "yes",
        },
    )
    transcript = prediction(
        "y",
        {
            "social_norm_topic_supported": "yes",
            "specific_hypothesis_supported": "yes",
            "demonstration_or_example_language_present": "yes",
            "situated_dialogue_or_action_cues_present": "yes",
            "discussion_or_advice_only": "no",
        },
    )

    with pytest.raises(ValueError, match="transcript identity mismatch"):
        evaluate(m, [video], [video], [transcript])


def test_strict_rule_requires_all_three_modalities_and_label_agreement():
    m = [manual("x", visual="yes", social="yes", label="yes")]
    video = prediction(
        "x",
        {
            "situated_social_scenario_visible": "yes",
            "proposed_norm_plausibly_demonstrated": "yes",
        },
    )
    transcript = prediction(
        "x",
        {
            "social_norm_topic_supported": "yes",
            "specific_hypothesis_supported": "yes",
            "demonstration_or_example_language_present": "yes",
            "situated_dialogue_or_action_cues_present": "yes",
            "discussion_or_advice_only": "no",
        },
    )

    report = evaluate(m, [video], [video], [transcript])

    metric = report["rules"]["strict_av_label_agreement"]["strict_current_label"]
    assert metric["tp"] == 1
    assert metric["precision"] == 1
    assert metric["recall"] == 1


def test_transcript_veto_does_not_treat_uncertain_as_negative():
    m = [manual("x", visual="yes", social="yes", label="yes")]
    video = prediction(
        "x",
        {
            "situated_social_scenario_visible": "yes",
            "proposed_norm_plausibly_demonstrated": "yes",
        },
    )
    transcript = prediction(
        "x",
        {
            "social_norm_topic_supported": "uncertain",
            "specific_hypothesis_supported": "uncertain",
            "demonstration_or_example_language_present": "uncertain",
            "situated_dialogue_or_action_cues_present": "uncertain",
            "discussion_or_advice_only": "no",
        },
    )

    report = evaluate(m, [video], [video], [transcript])

    assert report["rules"]["dual_scene_transcript_topic_veto"][
        "keep_after_relabel"
    ]["tp"] == 1


def test_v4_ensemble_uses_usable_after_relabel_and_social_domain():
    m = [manual("x", visual="yes", social="yes", label="no")]
    video = prediction(
        "x",
        {
            "usable_demo_after_relabel": "yes",
            "proposed_norm_supported": "no",
            "social_norm_domain": "yes",
        },
    )
    transcript = prediction(
        "x",
        {
            "social_norm_topic_supported": "uncertain",
            "specific_hypothesis_supported": "uncertain",
            "demonstration_or_example_language_present": "uncertain",
            "situated_dialogue_or_action_cues_present": "uncertain",
            "discussion_or_advice_only": "no",
        },
    )

    report = evaluate(
        m,
        [video],
        [video],
        [transcript],
        video_rubric="v4",
    )

    assert report["rules"]["dual_scene_dual_domain"]["keep_after_relabel"]["tp"] == 1
    assert report["rules"]["strict_av_label_agreement"]["strict_current_label"][
        "tn"
    ] == 1
