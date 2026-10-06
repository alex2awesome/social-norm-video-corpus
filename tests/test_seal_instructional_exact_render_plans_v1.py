import copy

import pytest

from scripts.seal_instructional_exact_render_plans_v1 import seal, template


def plan():
    return {
        "candidate_id": "c", "pillar": "instructional", "uid": "u",
        "proposed_start_sec": 1.0, "proposed_end_sec": 4.0,
        "transform_required": "manual_instructional_audio_policy_required",
        "ready_for_final_render": False,
        "approval_status": "unreviewed_localization_candidate",
    }


def review():
    return {
        "candidate_id": "c", "pillar": "instructional", "uid": "u",
        "proposed_start_sec": "1.0", "proposed_end_sec": "4.0",
        "exact_start_sec": "1.1", "exact_end_sec": "3.9",
        "visual_event_complete": "yes", "actor_target_grounded": "yes",
        "exact_behavior_label_supported": "yes", "start_boundary_clean": "yes",
        "end_boundary_clean": "yes", "label_bearing_visible_text_absent": "yes",
        "audio_content": "diegetic_behavior",
        "audio_is_required_for_behavior": "yes",
        "label_bearing_explanation_audio_overlaps": "no",
        "audio_action": "preserve", "manual_description": "One person apologizes.",
        "manual_rationale": "The apology is in-character and clean.",
        "ready_for_final_render": "yes",
    }


def test_template_keeps_label_and_audio_policy_unreviewed():
    row = template([plan()])[0]
    assert row["audio_action"] == ""
    assert row["exact_start_sec"] == ""


def test_preserves_required_clean_diegetic_audio():
    row = seal([plan()], [review()])[0]
    assert row["ready_for_final_render"] is True
    assert row["transform_required"] == "temporal_cut_preserve_audio"


def test_can_mute_label_audio_only_when_behavior_does_not_depend_on_it():
    value = review()
    value.update({
        "audio_content": "label_bearing_explanation",
        "audio_is_required_for_behavior": "no",
        "label_bearing_explanation_audio_overlaps": "yes",
        "audio_action": "mute",
    })
    row = seal([plan()], [value])[0]
    assert row["transform_required"] == "temporal_cut_and_mute"
    bad = copy.deepcopy(value); bad["audio_is_required_for_behavior"] = "yes"
    with pytest.raises(ValueError, match="cannot mute"):
        seal([plan()], [bad])


def test_mixed_required_and_label_audio_must_reject():
    value = review()
    value.update({
        "audio_content": "mixed_diegetic_and_explanation",
        "label_bearing_explanation_audio_overlaps": "yes",
        "audio_action": "reject", "ready_for_final_render": "no",
    })
    row = seal([plan()], [value])[0]
    assert row["ready_for_final_render"] is False
    assert row["transform_required"].startswith("rejected")


def test_clean_diegetic_audio_cannot_be_muted_for_convenience():
    value = review()
    value.update({
        "audio_is_required_for_behavior": "no", "audio_action": "mute",
    })
    with pytest.raises(ValueError, match="must not be muted"):
        seal([plan()], [value])


def test_label_audio_kind_must_match_overlap_judgment():
    value = review()
    value["audio_content"] = "label_bearing_explanation"
    value["audio_action"] = "reject"
    value["ready_for_final_render"] = "no"
    with pytest.raises(ValueError, match="contradict"):
        seal([plan()], [value])
