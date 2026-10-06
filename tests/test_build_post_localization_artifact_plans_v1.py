from scripts.build_post_localization_artifact_plans_v1 import (
    commentary_plan,
    instructional_plan,
    storyboard_bounds,
)


def test_storyboard_bounds_use_midpoints_not_claimed_exact_boundaries():
    assert storyboard_bounds([0.0, 1.0, 2.0, 3.0], 1, 2) == (0.5, 2.5)


def test_instructional_partial_demo_requires_manual_corrected_label():
    selection = [{
        "item_id": "i", "uid": "u", "norm": "wrong",
        "source_clip": "/source.mp4", "source_clip_sha256": "h",
    }]
    storyboards = [{"item_id": "i", "sampled_timestamps": [0.0, 1.0, 2.0]}]
    manual = [{
        "item_id": "i", "visual_demo": "yes", "label_alignment": "partial",
        "corrected_behavior_label": "one person returns another person's item",
        "segment_start_frame": "0", "segment_end_frame": "1",
    }]
    row = instructional_plan(selection, storyboards, manual)[0]
    assert row["behavior_label"].startswith("one person returns")
    assert row["ready_for_final_render"] is False
    assert row["bounds_basis"].endswith("needs_exact_video_audit")
    assert row["transform_required"] == "manual_instructional_audio_policy_required"


def test_commentary_clean_window_requires_mute_and_post_transform_audit():
    selected = [{
        "window_id": "w", "uid": "u", "source_path": "/source.mp4",
        "window_start_sec": 2.0, "window_end_sec": 14.0,
        "action_label": "one person takes another's bag",
    }]
    blind = [{
        "window_id": "w", "performed_event_visible": "yes",
        "actor_target_grounded": "yes", "before_action_after_complete": "yes",
        "crucial_action_occluded_or_offframe": "no",
        "label_bearing_text_absent": "yes",
    }]
    post = [{
        "window_id": "w", "label_alignment": "exact",
        "start_boundary_clean": "yes", "end_boundary_clean": "yes",
        "commentary_visual_route": "commentary_visual",
    }]
    row = commentary_plan(selected, blind, post)[0]
    assert row["ready_for_final_render"] is True
    assert row["transform_required"] == "temporal_cut_and_mute"
    assert row["approval_status"] == "unreviewed_localization_candidate"


def test_commentary_visible_label_text_routes_to_manual_crop_plan():
    selected = [{
        "window_id": "w", "uid": "u", "source_path": "/source.mp4",
        "window_start_sec": 2.0, "window_end_sec": 14.0,
        "action_label": "act",
    }]
    blind = [{
        "window_id": "w", "performed_event_visible": "yes",
        "actor_target_grounded": "yes", "before_action_after_complete": "yes",
        "crucial_action_occluded_or_offframe": "no",
        "label_bearing_text_absent": "no",
    }]
    post = [{
        "window_id": "w", "label_alignment": "exact",
        "start_boundary_clean": "yes", "end_boundary_clean": "yes",
        "commentary_visual_route": "commentary_visual",
    }]
    row = commentary_plan(selected, blind, post)[0]
    assert row["ready_for_final_render"] is False
    assert "crop_visible_label_text" in row["transform_required"]


def test_commentary_instructional_reroute_keeps_audio_policy_unresolved():
    selected = [{
        "window_id": "w", "uid": "u", "source_path": "/source.mp4",
        "window_start_sec": 2.0, "window_end_sec": 14.0,
        "action_label": "one person apologizes to another",
    }]
    blind = [{
        "window_id": "w", "performed_event_visible": "yes",
        "actor_target_grounded": "yes", "before_action_after_complete": "yes",
        "crucial_action_occluded_or_offframe": "no",
        "label_bearing_text_absent": "yes",
    }]
    post = [{
        "window_id": "w", "label_alignment": "exact",
        "start_boundary_clean": "yes", "end_boundary_clean": "yes",
        "commentary_visual_route": "instructional_demo",
    }]
    row = commentary_plan(selected, blind, post)[0]
    assert row["pillar"] == "instructional"
    assert row["ready_for_final_render"] is False
    assert row["transform_required"] == "manual_instructional_audio_policy_required"


def test_commentary_text_only_route_creates_no_visual_artifact_plan():
    selected = [{
        "window_id": "w", "uid": "u", "source_path": "/source.mp4",
        "window_start_sec": 2.0, "window_end_sec": 14.0,
        "action_label": "act",
    }]
    blind = [{
        "window_id": "w", "performed_event_visible": "yes",
        "actor_target_grounded": "yes", "before_action_after_complete": "yes",
        "crucial_action_occluded_or_offframe": "no",
        "label_bearing_text_absent": "yes",
    }]
    post = [{
        "window_id": "w", "label_alignment": "exact",
        "start_boundary_clean": "yes", "end_boundary_clean": "yes",
        "commentary_visual_route": "text_only",
    }]
    assert commentary_plan(selected, blind, post) == []
