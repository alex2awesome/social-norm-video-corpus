#!/usr/bin/env python3
"""Fail-closed shadow contract for turning text commentary into visual clips."""

from __future__ import annotations

from typing import Any


TRI = {"yes", "no", "uncertain"}
TEXT_ACCEPTS = {"accept", "accept_after_relabel"}
VISIBLE_OCCURRENCES = {"on_camera_visual_action", "on_camera_speech_act"}
CLEAR_DEMOS = {"clear_visual", "clear_audiovisual"}

ENUMS = {
    "text_label_decision": {"accept", "accept_after_relabel", "reject"},
    "behavior_occurrence": {
        "on_camera_visual_action",
        "on_camera_speech_act",
        "offscreen_only",
        "described_over_broll",
        "uncertain",
        "none",
    },
    "event_identity_grounded": TRI,
    "actor_target_or_context_grounded_in_clip": TRI,
    "label_span_relation": {
        "separate_before_behavior",
        "separate_after_behavior",
        "overlaps_behavior",
        "same_utterance",
        "not_found",
        "uncertain",
    },
    "action_clip_contains_normative_label_signal": TRI,
    "action_artifact_transform": {"none", "temporal_cut", "crop", "mute_and_crop"},
    "sanitization_verified": TRI,
    "demo_quality": {
        "clear_visual",
        "clear_audiovisual",
        "incomplete_or_ambiguous",
        "none",
        "uncertain",
    },
    "boundary_basis": {"exact_video", "frame_grid_only", "none"},
    "authenticity": {
        "organic",
        "organic_news_footage",
        "hidden_camera_genuine",
        "scripted",
        "animation",
        "news_or_commentary",
        "uncertain",
    },
}


def derive_commentary_visual_disposition(raw: dict[str, Any]) -> dict[str, Any]:
    """Derive whether a commentary text label can supervise a separate clip."""
    row = dict(raw)
    for field, allowed in ENUMS.items():
        if row.get(field) not in allowed:
            raise ValueError(f"invalid {field}: {row.get(field)!r}")

    if row["text_label_decision"] not in TEXT_ACCEPTS:
        row["disposition"] = "reject"
        return row
    if row["behavior_occurrence"] not in VISIBLE_OCCURRENCES:
        row["disposition"] = "uncertain" if row["behavior_occurrence"] == "uncertain" else "text_only"
        return row
    if row["event_identity_grounded"] != "yes" or row["actor_target_or_context_grounded_in_clip"] != "yes":
        row["disposition"] = "uncertain" if "uncertain" in {
            row["event_identity_grounded"], row["actor_target_or_context_grounded_in_clip"]
        } else "text_only"
        return row
    if row["label_span_relation"] not in {"separate_before_behavior", "separate_after_behavior"}:
        # An overlapping narrator/caption can only be removed when both the
        # audio and visible label regions were stripped and the resulting
        # artifact itself was manually re-audited.  A crop or mute alone is
        # insufficient.
        if not (
            row["label_span_relation"] in {"overlaps_behavior", "same_utterance"}
            and row["action_artifact_transform"] == "mute_and_crop"
            and row["sanitization_verified"] == "yes"
        ):
            row["disposition"] = "text_only"
            return row
    if row["action_clip_contains_normative_label_signal"] != "no":
        row["disposition"] = "uncertain" if row["action_clip_contains_normative_label_signal"] == "uncertain" else "text_only"
        return row
    if (
        row["action_artifact_transform"] != "none"
        and row["sanitization_verified"] != "yes"
    ):
        row["disposition"] = "uncertain"
        return row
    if row["demo_quality"] not in CLEAR_DEMOS or row["boundary_basis"] != "exact_video":
        row["disposition"] = "text_only"
        return row

    start = row.get("action_start_sec")
    end = row.get("action_end_sec")
    if start is None or end is None or not 0 <= float(start) < float(end):
        raise ValueError("exact commentary visual recovery requires ordered action bounds")

    if row["authenticity"] in {"scripted", "animation"}:
        row["disposition"] = "reroute_instructional_review"
    elif row["authenticity"] in {
        "organic",
        "organic_news_footage",
        "hidden_camera_genuine",
    }:
        row["disposition"] = "recover_commentary_visual"
    else:
        row["disposition"] = "uncertain" if row["authenticity"] == "uncertain" else "text_only"
    return row
