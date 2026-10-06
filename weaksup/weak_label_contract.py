#!/usr/bin/env python3
"""Deterministic derivations for auditable social-norm weak labels.

Models and human reviewers label the atomic fields below.  Composite fields are
calculated here; a model never gets to make an independent composite decision.
"""

from __future__ import annotations

from typing import Any


TRI = ("yes", "no", "uncertain")

ACTOR_KINDS = (
    "person",
    "human_group",
    "institution",
    "nonhuman",
    "none",
    "uncertain",
)

BEHAVIOR_KINDS = (
    "physical_action",
    "speech_act",
    "omission",
    "institutional_action",
    "technical_procedure",
    "accident_or_involuntary_event",
    "abstract_topic_or_trait",
    "none",
    "uncertain",
)

AFFECTED_CONTEXT_KINDS = (
    "person",
    "human_group",
    "shared_social_context",
    "private_self_only",
    "nonhuman_or_object_only",
    "none",
    "uncertain",
)

EXPECTATION_KINDS = (
    "interpersonal_treatment",
    "shared_coordination",
    "civic_or_institutional_duty",
    "conventional_role_obligation",
    "technical_correctness",
    "legal_rule_only",
    "health_or_physical_safety_only",
    "personal_preference",
    "none",
    "uncertain",
)


ATOMIC_SCOPE_SCHEMA: dict[str, dict[str, Any]] = {
    "actor_kind": {"type": "string", "enum": list(ACTOR_KINDS)},
    "behavior_kind": {"type": "string", "enum": list(BEHAVIOR_KINDS)},
    "affected_context_kind": {
        "type": "string",
        "enum": list(AFFECTED_CONTEXT_KINDS),
    },
    "expectation_kind": {"type": "string", "enum": list(EXPECTATION_KINDS)},
}


def _require(value: str, allowed: tuple[str, ...], field: str) -> None:
    if value not in allowed:
        raise ValueError(f"invalid {field}: {value!r}")


def derive_social_actor_grounded(actor_kind: str) -> str:
    """Whether supplied evidence identifies a human social actor."""
    _require(actor_kind, ACTOR_KINDS, "actor_kind")
    if actor_kind in {"person", "human_group", "institution"}:
        return "yes"
    if actor_kind in {"nonhuman", "none"}:
        return "no"
    return "uncertain"


def derive_concrete_behavior(behavior_kind: str) -> str:
    """Whether supplied evidence identifies a temporally concrete behavior.

    Technical actions and accidents can be concrete even though they are outside
    the target social-norm scope.  Scope is derived separately below.
    """
    _require(behavior_kind, BEHAVIOR_KINDS, "behavior_kind")
    if behavior_kind in {
        "physical_action",
        "speech_act",
        "omission",
        "institutional_action",
        "technical_procedure",
        "accident_or_involuntary_event",
    }:
        return "yes"
    if behavior_kind in {"abstract_topic_or_trait", "none"}:
        return "no"
    return "uncertain"


def derive_target_or_shared_context_grounded(affected_context_kind: str) -> str:
    """Whether supplied evidence identifies an affected human/shared setting."""
    _require(
        affected_context_kind,
        AFFECTED_CONTEXT_KINDS,
        "affected_context_kind",
    )
    if affected_context_kind in {"person", "human_group", "shared_social_context"}:
        return "yes"
    if affected_context_kind in {"private_self_only", "nonhuman_or_object_only", "none"}:
        return "no"
    return "uncertain"


def derive_is_social_norm(
    *,
    actor_kind: str,
    behavior_kind: str,
    affected_context_kind: str,
    expectation_kind: str,
) -> str:
    """Derive the in-scope social-norm gate from four atomic classifications.

    ``no`` means positively outside scope.  Missing or unresolved evidence yields
    ``uncertain`` and therefore cannot pass a fail-closed acceptance gate.
    """
    _require(actor_kind, ACTOR_KINDS, "actor_kind")
    _require(behavior_kind, BEHAVIOR_KINDS, "behavior_kind")
    _require(
        affected_context_kind,
        AFFECTED_CONTEXT_KINDS,
        "affected_context_kind",
    )
    _require(expectation_kind, EXPECTATION_KINDS, "expectation_kind")

    outside_scope = (
        actor_kind == "nonhuman"
        or behavior_kind
        in {
            "technical_procedure",
            "accident_or_involuntary_event",
            "abstract_topic_or_trait",
        }
        or affected_context_kind in {"private_self_only", "nonhuman_or_object_only"}
        or expectation_kind
        in {
            "technical_correctness",
            "legal_rule_only",
            "health_or_physical_safety_only",
            "personal_preference",
            "none",
        }
    )
    if outside_scope:
        return "no"

    unresolved = (
        actor_kind in {"none", "uncertain"}
        or behavior_kind in {"none", "uncertain"}
        or affected_context_kind in {"none", "uncertain"}
        or expectation_kind == "uncertain"
    )
    if unresolved:
        return "uncertain"

    return "yes"


def derive_scope_labels(result: dict[str, Any]) -> dict[str, str]:
    """Return every derived scope field for a model/manual atomic result."""
    return {
        "social_actor_grounded": derive_social_actor_grounded(result["actor_kind"]),
        "concrete_behavior": derive_concrete_behavior(result["behavior_kind"]),
        "target_or_shared_context_grounded": derive_target_or_shared_context_grounded(
            result["affected_context_kind"]
        ),
        "is_social_norm": derive_is_social_norm(
            actor_kind=result["actor_kind"],
            behavior_kind=result["behavior_kind"],
            affected_context_kind=result["affected_context_kind"],
            expectation_kind=result["expectation_kind"],
        ),
    }


def attach_scope_labels(result: dict[str, Any]) -> dict[str, Any]:
    """Copy an atomic result and attach code-derived composite fields."""
    return {**result, **derive_scope_labels(result)}
