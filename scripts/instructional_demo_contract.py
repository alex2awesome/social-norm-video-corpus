#!/usr/bin/env python3
"""Deterministic, format-agnostic contract for instructional demonstrations.

Models and reviewers supply atomic observations.  This module derives a
review disposition; it never asks a model to decide whether a corpus item is
kept.  Scripted live action, animation, puppets, games, and hidden-camera
setups may all qualify when they depict one connected, label-aligned social
behavior.  Explanation polarity is not itself a rejection condition.
"""

from __future__ import annotations

from typing import Any

from weak_label_contract import attach_scope_labels


TRI = {"yes", "no", "uncertain"}
PERFORMED_MEDIA = {
    "live_action",
    "animation",
    "puppet_or_toy",
    "game_or_simulation",
    "hidden_camera_or_social_experiment",
}
CLEAR_DEMOS = {"clear_visual", "clear_audiovisual"}

ENUMS = {
    "performed_behavior": TRI,
    "actor_grounded_in_demo": TRI,
    "target_or_social_context_grounded_in_demo": TRI,
    "connected_episode": TRI,
    "label_alignment": {"exact", "repairable", "mismatch", "uncertain"},
    "demo_medium": PERFORMED_MEDIA
    | {
        "talking_head_only",
        "text_or_graphic_only",
        "generic_broll",
        "screen_tutorial",
        "audio_only",
        "uncertain",
    },
    "demo_quality": CLEAR_DEMOS
    | {"incomplete_or_ambiguous", "none", "uncertain"},
    "boundary_basis": {"exact_video", "frame_grid_only", "none"},
    "polarity": {"violation", "correct", "contrast", "explanation", "uncertain"},
}


def _validate(row: dict[str, Any]) -> None:
    for field, allowed in ENUMS.items():
        if row.get(field) not in allowed:
            raise ValueError(f"invalid {field}: {row.get(field)!r}")


def _ordered_bounds(row: dict[str, Any]) -> bool:
    start = row.get("demo_start_sec")
    end = row.get("demo_end_sec")
    return (
        start is not None
        and end is not None
        and 0 <= float(start) < float(end)
    )


def derive_instructional_demo_disposition(raw: dict[str, Any]) -> dict[str, Any]:
    """Derive a non-destructive instructional review disposition.

    ``instructional_demo_candidate`` is still a review route, not automatic
    acceptance.  ``*_after_relabel`` preserves a useful depicted behavior
    whose original weak label is wrong.  Sparse-frame evidence may nominate a
    source for recutting, but cannot certify an exact training clip.
    """
    row = attach_scope_labels(dict(raw))
    _validate(row)

    if row["is_social_norm"] == "no":
        row["disposition"] = "out_of_scope_review"
        return row
    if row["is_social_norm"] == "uncertain":
        row["disposition"] = "uncertain"
        return row

    atoms = (
        row["performed_behavior"],
        row["actor_grounded_in_demo"],
        row["target_or_social_context_grounded_in_demo"],
        row["connected_episode"],
    )
    if "uncertain" in atoms or row["demo_medium"] == "uncertain":
        row["disposition"] = "uncertain"
        return row
    if (
        "no" in atoms
        or row["demo_medium"] not in PERFORMED_MEDIA
        or row["demo_quality"] not in CLEAR_DEMOS
    ):
        row["disposition"] = "text_only_instructional"
        return row

    if row["label_alignment"] == "uncertain":
        row["disposition"] = "uncertain"
        return row

    if row["boundary_basis"] != "exact_video" or not _ordered_bounds(row):
        row["disposition"] = "instructional_source_needs_recut"
        return row

    if row["label_alignment"] == "exact":
        row["disposition"] = "instructional_demo_candidate"
    else:
        # A mismatch is not discarded when the pixels still contain a clear
        # social demonstration; it must receive a scene-grounded relabel.
        row["disposition"] = "instructional_demo_candidate_after_relabel"
    return row

