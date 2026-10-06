import numpy as np

from scripts.evaluate_witnessed_reaction_external_holdout import (
    evaluate_frozen_rules,
    manual_target,
    metric_record,
    text_label,
    vlm_label,
)


def manual_row(role="bystander", temporal="action_established_before_reaction"):
    return {
        "reaction_source_role": role,
        "reaction_grounding": "visible_on_scene",
        "reaction_content": "protective_intervention",
        "temporal_relation": temporal,
    }


def vlm_row(role="bystander"):
    return {
        "reaction_visible_or_audibly_grounded": "yes",
        "reaction_source_role": role,
        "reaction_content": "protective_intervention",
        "reaction_targets_action": "yes",
        "action_established_before_reaction": "yes",
    }


def test_manual_target_keeps_strict_bystander_and_overlap():
    assert manual_target(manual_row(), include_authority=False)
    assert manual_target(
        manual_row(temporal="action_only_overlaps_reaction"),
        include_authority=False,
    )


def test_manual_target_reports_authority_separately():
    row = manual_row(role="authority_or_host")
    assert not manual_target(row, include_authority=False)
    assert manual_target(row, include_authority=True)


def test_manual_target_rejects_affected_target_and_generic_affect():
    row = manual_row(role="affected_target")
    assert not manual_target(row, include_authority=True)
    row = manual_row()
    row["reaction_content"] = "generic_affect"
    assert not manual_target(row, include_authority=True)


def test_vlm_label_requires_every_atomic_condition():
    assert vlm_label(vlm_row(), include_authority=False) == 1
    for key in (
        "reaction_visible_or_audibly_grounded",
        "reaction_targets_action",
        "action_established_before_reaction",
    ):
        row = vlm_row()
        row[key] = "no"
        assert vlm_label(row, include_authority=False) == 0


def test_text_label_detects_active_intervention_but_not_reported_description():
    assert text_label("Stop. Leave her alone.", "", "") == 1
    assert text_label("Stop him", "the video shows the incident", "") == 0
    assert text_label("Wow!", "", "") == 0


def test_frozen_vote_requires_two_signals_and_a_vlm():
    rows = [
        {
            "item_id": "positive",
            "reaction": "Stop that",
            "context": "",
            "title": "",
            "manual": manual_row(),
            "qwen_result": vlm_row(),
            "glm_result": {},
        },
        {
            "item_id": "text_only",
            "reaction": "Stop that",
            "context": "",
            "title": "",
            "manual": manual_row(role="affected_target"),
            "qwen_result": {},
            "glm_result": {},
        },
    ]
    report, errors = evaluate_frozen_rules(rows, include_authority=False)
    metric = report["metrics"]["two_of_three_with_vlm"]
    assert metric["predicted_positive"] == 1
    assert metric["true_positive"] == 1
    assert errors == []


def test_metric_record_uses_half_threshold():
    result = metric_record(
        np.asarray([1, 0, 1]), np.asarray([0.9, 0.8, 0.1])
    )
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
