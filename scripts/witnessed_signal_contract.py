#!/usr/bin/env python3
"""Deterministic shadow contract for witnessed weak supervision.

The reviewer/model supplies enumerable observations.  Code derives the social
scope and final disposition; a model is never asked whether an item should be
kept.  This module is intentionally production-independent.
"""

from __future__ import annotations

from typing import Any

from weak_label_contract import attach_scope_labels


TRI = {"yes", "no", "uncertain"}
# The 2026-07-24 fresh-query audit found that authority/host utterances were
# commonly police commands or arrest language.  Those can localize a useful
# visual event, but they do not independently certify the organic witnessed
# weak-supervision signal.  Preserve them for a separate scene-candidate route.
ALLOWED_REACTION_ROLES = {"bystander", "organic_audience"}
ALLOWED_REACTION_GROUNDING = {"visible_on_scene", "audible_on_scene"}
ALLOWED_REACTION_CONTENT = {
    "targeted_objection",
    "correction_or_sanction",
    "protective_intervention",
    "interposition_or_separation",
}
ALLOWED_DEMO_QUALITY = {"clear_visual", "clear_audiovisual"}
ORGANIC_AUTHENTICITY = {"organic", "hidden_camera_genuine"}
SCRIPTED_AUTHENTICITY = {"scripted", "animation"}

ENUMS = {
    "action_observed": TRI,
    "reaction_source_role": {
        "bystander",
        "authority_or_host",
        "organic_audience",
        "affected_target",
        "violator_or_actor",
        "narrator_or_commentator",
        "uncertain",
        "none",
    },
    "reaction_grounding": {
        "visible_on_scene",
        "audible_on_scene",
        "voiceover_or_remote",
        "uncertain",
        "none",
    },
    "reaction_content": {
        "targeted_objection",
        "correction_or_sanction",
        "protective_intervention",
        "interposition_or_separation",
        "generic_affect",
        "self_defense_or_excuse",
        "compliance_or_apology",
        "description_only",
        "uncertain",
        "none",
    },
    "temporal_relation": {
        "action_established_before_reaction",
        "action_only_overlaps_reaction",
        "reaction_before_action",
        "action_not_observed",
        "uncertain",
    },
    "reaction_candidate_origin": {
        "detector_selected",
        "nearby_transcript_recovery",
        "none",
        "uncertain",
    },
    "current_label_relation": {"exact", "repairable", "mismatch", "uncertain"},
    "pre_reaction_demo_quality": {
        "clear_visual",
        "clear_audiovisual",
        "incomplete_or_ambiguous",
        "contains_reaction_signal",
        "none",
        "uncertain",
    },
    "boundary_basis": {"exact_video", "frame_grid_only", "none"},
    "authenticity": {
        "organic",
        "hidden_camera_genuine",
        "scripted",
        "animation",
        "news_or_commentary",
        "uncertain",
    },
}


def _validate_enums(row: dict[str, Any]) -> None:
    for field, allowed in ENUMS.items():
        value = row.get(field)
        if value not in allowed:
            raise ValueError(f"invalid {field}: {value!r}")


def reaction_sequence_supported(row: dict[str, Any]) -> bool:
    """Return whether atomic evidence supports the strict witnessed signal."""
    return (
        row["reaction_source_role"] in ALLOWED_REACTION_ROLES
        and row["reaction_grounding"] in ALLOWED_REACTION_GROUNDING
        and row["reaction_content"] in ALLOWED_REACTION_CONTENT
        and row["temporal_relation"] == "action_established_before_reaction"
    )


def exact_splice_supported(row: dict[str, Any]) -> bool:
    """Require exact-video bounds; frame grids are discovery evidence only."""
    if row["pre_reaction_demo_quality"] not in ALLOWED_DEMO_QUALITY:
        return False
    if row["boundary_basis"] != "exact_video":
        return False
    action_end = row.get("action_end_sec")
    reaction_start = row.get("reaction_start_sec")
    return (
        action_end is not None
        and reaction_start is not None
        and 0 < float(action_end) <= float(reaction_start)
    )


def derive_witnessed_disposition(raw: dict[str, Any]) -> dict[str, Any]:
    """Attach scope labels and derive a fail-closed shadow disposition.

    ``strict_accept`` means the current detector-selected reaction and current
    label are supported. ``recover_strict`` means the source is useful only
    after selecting a nearby reaction and/or repairing the semantic label.
    Scripted material never becomes witnessed; it is sent to a separate
    instructional review rather than admitted here.
    """
    row = attach_scope_labels(dict(raw))
    title_staging_cue = row.get("source_staging_title_cue", False)
    if not isinstance(title_staging_cue, bool):
        raise ValueError(
            f"invalid source_staging_title_cue: {title_staging_cue!r}"
        )
    authority_text_cue = row.get("authority_reaction_text_cue", False)
    if not isinstance(authority_text_cue, bool):
        raise ValueError(
            f"invalid authority_reaction_text_cue: {authority_text_cue!r}"
        )
    _validate_enums(row)

    if row["action_observed"] == "no" or row["is_social_norm"] == "no":
        row["disposition"] = "reject"
        return row

    if row["action_observed"] == "uncertain" or row["is_social_norm"] == "uncertain":
        row["disposition"] = "uncertain"
        return row

    # Independently confirmed on 48 unseen sources: an explicit
    # prank/social-experiment/hidden-camera title was 32/32 precise for a
    # deliberately produced setup. Preserve the source, but do not let it
    # certify an organic witnessed item. Instructional review separately
    # decides whether the pixels and semantic label form a usable demo.
    if title_staging_cue:
        row["production_style"] = "staged_prank_candidate"
        row["staging_rule_id"] = (
            "title_prank_social_experiment_hidden_camera_v1"
        )
        row["disposition"] = "reroute_instructional_review"
        return row

    # Exact-span authority/enforcement/host reaction cues were independently
    # confirmed on 96 source-disjoint clips: v3 produced 33 true positives and
    # 0 false positives across discovery + confirmation audits. Such text can
    # localize a useful scene, but it cannot certify an organic bystander
    # reaction. Preserve the clip as a scene candidate.
    if authority_text_cue:
        row["disposition"] = (
            "scene_candidate"
            if row["pre_reaction_demo_quality"] in ALLOWED_DEMO_QUALITY
            and row["current_label_relation"] in {"exact", "repairable"}
            else "reject"
        )
        row["strict_witnessed_exclusion_reason"] = (
            "authority_reaction_text_cue_v3"
        )
        return row

    if row["authenticity"] in SCRIPTED_AUTHENTICITY:
        row["disposition"] = (
            "reroute_instructional_review"
            if row["pre_reaction_demo_quality"] in ALLOWED_DEMO_QUALITY
            and row["current_label_relation"] in {"exact", "repairable"}
            else "reject"
        )
        return row

    if row["authenticity"] not in ORGANIC_AUTHENTICITY:
        row["disposition"] = "uncertain" if row["authenticity"] == "uncertain" else "reject"
        return row

    if row["reaction_source_role"] == "authority_or_host":
        row["disposition"] = (
            "scene_candidate"
            if row["pre_reaction_demo_quality"] in ALLOWED_DEMO_QUALITY
            and row["current_label_relation"] in {"exact", "repairable"}
            else "reject"
        )
        row["strict_witnessed_exclusion_reason"] = "authority_or_host_reaction"
        return row

    if not reaction_sequence_supported(row) or not exact_splice_supported(row):
        row["disposition"] = "reject"
        return row

    if row["reaction_candidate_origin"] == "detector_selected" and row["current_label_relation"] == "exact":
        row["disposition"] = "strict_accept"
    elif (
        row["reaction_candidate_origin"] in {"detector_selected", "nearby_transcript_recovery"}
        and row["current_label_relation"] in {"exact", "repairable"}
    ):
        row["disposition"] = "recover_strict"
    else:
        row["disposition"] = "reject"
    return row
