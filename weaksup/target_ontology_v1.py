#!/usr/bin/env python3
"""Versioned revised target ontology for weak supervision (roadmap section 2.3).

This module freezes the revised target decomposition: witnessed validity no
longer requires independent-bystander identity (that subtype is preserved as an
auxiliary target), audiovisual grounding distinguishes visible physical action
from situated speech from description-only, and every final target is
deterministically derived from enumerated atomic fields.  Nothing here inspects
media or creates an acceptance label.
"""

from __future__ import annotations

from typing import Any

try:
    from weaksup.weak_label_contract import derive_is_social_norm
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from weaksup.weak_label_contract import derive_is_social_norm


ONTOLOGY_VERSION = "target_ontology_v1"

TRI = ("yes", "no", "uncertain")

PILLARS = ("witnessed", "instructional", "commentary")

# Shared revised targets (roadmap section 2.3).
SHARED_TARGETS: dict[str, str] = {
    "norm_event_supported": (
        "the evidence supports a concrete social behavior and a socially "
        "legible expectation, judgment, or response"
    ),
    "event_audiovisually_grounded": (
        "the behavior is visible, or is a situated speech act whose "
        "participants and context are audiovisually present in the saved "
        "interval; narration describing an absent event never qualifies"
    ),
    "label_alignment": (
        "the assigned normalized behavior, norm, and polarity match the event"
    ),
    "reaction_grounded": (
        "an observable response is temporally and causally connected to the "
        "behavior"
    ),
    "independent_bystander_signal": (
        "the responder is a distinct third party or organic audience member; "
        "a valuable witnessed subtype, not the universal definition of "
        "whether a norm event exists"
    ),
    "clean_training_span": (
        "the event interval can be separated from the textual, spoken, "
        "graphic, or reaction signal that supplied the label"
    ),
}

# Per-pillar label-model targets (roadmap section 12.4).
PILLAR_TARGETS: dict[str, tuple[str, ...]] = {
    "witnessed": (
        "norm_event_supported",
        "reaction_grounded",
        "independent_bystander_signal",
        "event_audiovisually_grounded",
        "clean_pre_reaction_span",
    ),
    "instructional": (
        "explicit_social_rule_grounded",
        "demonstration_present",
        "event_audiovisually_grounded",
        "label_alignment",
        "clean_demo_span",
    ),
    "commentary": (
        "commentary_text_label_supported",
        "occurred_event_supported",
        "event_present_in_source",
        "event_audiovisually_grounded",
        "clean_action_span",
    ),
}

# How the saved interval grounds the event.  ``description_only`` covers
# narration/explanation of an absent event; it is never audiovisual grounding.
GROUNDING_MODES = (
    "physical_action_visible",
    "situated_speech_audiovisual",
    "description_only",
    "absent",
    "uncertain",
)

# Responder provenance roles.  Non-bystander roles can still support
# ``norm_event_supported``; they only fail the strict organic subtype.
RESPONDER_ROLES = (
    "independent_bystander",
    "organic_audience",
    "affected_party",
    "participant",
    "camera_operator",
    "host",
    "authority",
    "unknown",
)

BYSTANDER_ROLES = {"independent_bystander", "organic_audience"}

LABEL_RELATIONS = (
    "exact",
    "broad_but_correct",
    "wrong_but_relabelable",
    "unsupported",
    "uncertain",
)


def _require(value: Any, allowed: tuple[str, ...], field: str) -> None:
    if value not in allowed:
        raise ValueError(f"invalid {field}: {value!r}")


def tri_and(*values: str) -> str:
    """Fail-closed tri-state conjunction: no > uncertain > yes."""
    for value in values:
        _require(value, TRI, "tri value")
    if "no" in values:
        return "no"
    if "uncertain" in values:
        return "uncertain"
    return "yes"


def derive_event_audiovisually_grounded(grounding_mode: str) -> str:
    _require(grounding_mode, GROUNDING_MODES, "grounding_mode")
    if grounding_mode in {"physical_action_visible", "situated_speech_audiovisual"}:
        return "yes"
    if grounding_mode in {"description_only", "absent"}:
        return "no"
    return "uncertain"


def derive_norm_event_supported(
    *,
    actor_kind: str,
    behavior_kind: str,
    affected_context_kind: str,
    expectation_kind: str,
    normative_signal_grounded: str,
) -> str:
    """A concrete in-scope social behavior plus any socially legible signal.

    ``normative_signal_grounded`` is yes when a reaction, stance, or explicit
    rule is grounded — from any responder role.  Independent-bystander identity
    is deliberately not required here.
    """
    scope = derive_is_social_norm(
        actor_kind=actor_kind,
        behavior_kind=behavior_kind,
        affected_context_kind=affected_context_kind,
        expectation_kind=expectation_kind,
    )
    return tri_and(scope, normative_signal_grounded)


def derive_reaction_grounded(
    *,
    response_observable: str,
    response_after_or_overlaps: str,
    response_targets_action: str,
) -> str:
    return tri_and(
        response_observable, response_after_or_overlaps, response_targets_action
    )


def derive_independent_bystander_signal(
    *, reaction_grounded: str, responder_role: str
) -> str:
    """Strict organic subtype.  A ``no`` here must never be treated as
    evidence against ``norm_event_supported``; it only excludes the item from
    the strict independent-bystander subcorpus."""
    _require(reaction_grounded, TRI, "reaction_grounded")
    _require(responder_role, RESPONDER_ROLES, "responder_role")
    if reaction_grounded == "no":
        return "no"
    if responder_role == "unknown":
        return "uncertain"
    if responder_role not in BYSTANDER_ROLES:
        return "no"
    return reaction_grounded


def derive_label_alignment(label_relation: str) -> str:
    _require(label_relation, LABEL_RELATIONS, "label_relation")
    if label_relation in {"exact", "broad_but_correct"}:
        return "yes"
    if label_relation == "unsupported":
        return "no"
    return "uncertain"


def label_disposition(label_relation: str) -> str:
    """Repairable labels are preserved for relabel review, never rejected."""
    _require(label_relation, LABEL_RELATIONS, "label_relation")
    return {
        "exact": "aligned",
        "broad_but_correct": "aligned",
        "wrong_but_relabelable": "repairable_relabel_review",
        "unsupported": "unsupported_review",
        "uncertain": "uncertain_review",
    }[label_relation]


def derive_clean_training_span(
    *, bounds_valid: str, label_signal_excluded: str
) -> str:
    return tri_and(bounds_valid, label_signal_excluded)


def pillar_targets(pillar: str) -> tuple[str, ...]:
    _require(pillar, PILLARS, "pillar")
    return PILLAR_TARGETS[pillar]
