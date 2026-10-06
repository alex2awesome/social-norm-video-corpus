import pytest

from scripts.evaluate_manual_canary_predictions import (
    build_report,
    gold_labels,
    positive_score,
)


def manifest_row(item_id="instructional:x:0", ordinal=0):
    return {"item_id": item_id, "uid": "x", "ordinal": ordinal}


def review_row(item_id="instructional:x:0", ordinal=0):
    return {
        "item_id": item_id,
        "uid": "x",
        "ordinal": ordinal,
        "decision": "accept",
        "evidence": "acted exchange",
        "visual_demo_present": "yes",
        "is_social_norm": "yes",
        "norm_supported": "yes",
    }


def prediction_row(item_id="instructional:x:0"):
    return {
        "item_id": item_id,
        "error": None,
        "result": {
            "target_pillar_signal_visible": "yes",
            "social_behavior_scene_visible": "yes",
            "confidence": 0.9,
            "evidence": "two people interact",
        },
    }


def test_gold_separates_relabel_rescue_from_strict_current_label():
    row = review_row()
    row["norm_supported"] = "no"

    assert gold_labels(row) == {
        "keep_after_relabel": True,
        "strict_current_label": False,
    }


def test_positive_score_is_conservative_for_uncertain():
    assert positive_score("yes", 0.9) == 0.9
    assert positive_score("no", 0.9) == pytest.approx(0.1)
    assert positive_score("uncertain", 0.99) == 0.5


def test_complete_report_scores_prediction_and_preserves_disagreement_evidence():
    report = build_report(
        [manifest_row()],
        [review_row()],
        [prediction_row()],
        prediction_keys=["target_pillar_signal_visible"],
        require_complete=True,
    )

    score = report["tasks"]["strict_current_label"][
        "target_pillar_signal_visible"
    ][0]
    assert (score["tp"], score["fp"], score["fn"]) == (1, 0, 0)
    assert score["precision"] == 1.0
    assert report["manual_gold_counts"]["keep_after_relabel"] == 1


def test_report_rejects_partial_manual_review_and_duplicate_predictions():
    with pytest.raises(ValueError, match="cover the manifest exactly"):
        build_report(
            [manifest_row(), manifest_row("instructional:y:0", 1)],
            [review_row()],
            [prediction_row()],
            prediction_keys=["target_pillar_signal_visible"],
            require_complete=False,
        )
    with pytest.raises(ValueError, match="duplicate item_id"):
        build_report(
            [manifest_row()],
            [review_row()],
            [prediction_row(), prediction_row()],
            prediction_keys=["target_pillar_signal_visible"],
            require_complete=False,
        )
