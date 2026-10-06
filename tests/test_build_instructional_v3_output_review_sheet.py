import pytest

from scripts.build_instructional_v3_output_review_sheet import apply_review_overlay, build_rows


def test_build_rows_exposes_gold_prediction_and_rationale() -> None:
    rows = build_rows(
        [{"storyboard_index": 2, "item_id": "clip"}],
        [{
            "audit_index": "2",
            "manual_visual_demo": "yes",
            "visual_form": "roleplay",
            "manual_evidence": "Two people perform a refusal.",
        }],
        [{
            "item_id": "clip",
            "error": None,
            "result": {
                "demo_candidate": True,
                "visual_sequence": "roleplay_or_fiction",
                "participant_configuration": "visible_interacting_participants",
                "behavior_evidence": "visibly_performed",
                "specific_social_act": "yes",
                "literal_behavior": "One person refuses another.",
                "evidence": "Both participants and refusal are visible.",
            },
        }],
    )
    assert rows[0]["manual_visual_demo"] == "yes"
    assert rows[0]["v3_demo_candidate"] == "yes"
    assert rows[0]["output_reviewed"] == ""
    assert rows[0]["v3_literal_behavior"] == "One person refuses another."


def test_build_rows_fails_without_successful_output_coverage() -> None:
    with pytest.raises(ValueError, match="successful coverage mismatch"):
        build_rows(
            [{"storyboard_index": 2, "item_id": "clip"}],
            [{"audit_index": "2", "manual_visual_demo": "yes"}],
            [],
        )


def test_apply_review_overlay_requires_complete_human_review() -> None:
    source = [{"audit_index": 2, "output_reviewed": ""}]
    merged = apply_review_overlay(source, [{
        "audit_index": "2",
        "output_reviewed": "yes",
        "output_visually_supported": "no",
        "error_mechanism": "static_overcall",
        "note": "No behavior unfolds.",
    }])
    assert merged[0]["output_reviewed"] == "yes"
    assert merged[0]["output_visually_supported"] == "no"


def test_apply_review_overlay_fails_on_missing_or_unreviewed_rows() -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        apply_review_overlay([{"audit_index": 2}], [])
    with pytest.raises(ValueError, match="output not reviewed"):
        apply_review_overlay([{"audit_index": 2}], [{
            "audit_index": "2",
            "output_reviewed": "no",
            "output_visually_supported": "no",
        }])
