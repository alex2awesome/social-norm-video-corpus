#!/usr/bin/env python3
"""Fail-closed contract for commentary event-label model outputs.

This is a shadow contract, not an automatic corpus filter.  It operationalizes
the five failure modes found by complete second adjudication of the fresh
30-source commentary cohort.
"""

from __future__ import annotations

from typing import Any


TRI = {"yes", "no", "uncertain"}
EVENT_SCOPE = {
    "bounded_occurrence",
    "repeated_specific_occurrence",
    "generic_or_hypothetical",
    "aggregate_examples",
    "ambiguous_or_unresolved",
    "no_candidate_event",
}
STANCE_QUALITY = {
    "unambiguous_external",
    "contemporaneous_bystander",
    "affected_party_objection",
    "contested",
    "missing_or_unresolved",
}
ROUTES = {
    "strict_visual_search",
    "instructional_definition_retrieval",
    "exploratory_contested",
    "exploratory_ambiguous",
    "none",
}
STRICT_SCOPE = {"bounded_occurrence", "repeated_specific_occurrence"}
STRICT_STANCE = {
    "unambiguous_external",
    "contemporaneous_bystander",
    "affected_party_objection",
}
ATOMIC_YES = (
    "is_social_norm",
    "occurred_event_grounded",
    "social_actor_grounded",
    "behavior_semantically_specific",
    "target_or_shared_context_grounded",
    "normative_stance_grounded",
)
REQUIRED = set(ATOMIC_YES) | {
    "event_scope",
    "stance_quality",
    "behavior_evidence_quote",
    "stance_evidence_quote",
    "normalized_behavior",
    "normalized_norm",
    "route",
    "description",
}


SYSTEM_PROMPT = """You are auditing a transcript-derived commentary candidate for
weak supervision of a visible social event. Use transcript evidence only; title,
query, category, and prior labels are retrieval hints and cannot fill missing
evidence.

A strict event candidate requires a particular occurred actor-action-target/shared
context. The action must name observable conduct, not merely an evaluation such as
'acting badly', 'being rude', or 'driving like a jerk'. The transcript must also
ground a normative stance and its source. A generic definition or hypothetical may
be routed only as an instructional-demo retrieval lead. Aggregate examples without
one event anchor, unresolved accusation/denial, and explicit disagreement about the
norm must not receive an unequivocal positive event label. Preserve contested and
ambiguous cases in their named exploratory routes; never silently discard them.

The transcript proposes semantics and temporal anchors but never certifies that the
event is visible. Strict candidates still require complete blind audiovisual review."""


ATOMIC_SCOPE_FIELDS = (
    "actor_kind",
    "behavior_kind",
    "affected_context_kind",
    "expectation_kind",
)


def validate_result_atomic(result: dict[str, Any]) -> dict[str, Any]:
    """Atomic-scope variant: ``is_social_norm`` is derived, never supplied.

    The legacy contract accepted ``is_social_norm`` as a direct model input,
    which invited treating an estimate of the latent target as an independent
    observation.  This wrapper requires the four shared atomic scope fields,
    derives the composite in code, and rejects a contradictory supplied value.
    """
    try:
        from scripts.weak_label_contract import derive_is_social_norm
    except ModuleNotFoundError:  # pragma: no cover - direct script execution
        from weak_label_contract import derive_is_social_norm

    missing = [field for field in ATOMIC_SCOPE_FIELDS if field not in result]
    if missing:
        raise ValueError(f"missing atomic scope fields: {missing}")
    derived = derive_is_social_norm(
        **{field: result[field] for field in ATOMIC_SCOPE_FIELDS}
    )
    supplied = result.get("is_social_norm")
    if supplied is not None and supplied != derived:
        raise ValueError(
            f"supplied is_social_norm {supplied!r} contradicts atomic derivation {derived!r}"
        )
    return validate_result({**result, "is_social_norm": derived})


def validate_result(result: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED - set(result)
    if missing:
        raise ValueError(f"missing commentary v3 fields: {sorted(missing)}")
    for field in ATOMIC_YES:
        if result[field] not in TRI:
            raise ValueError(f"{field} must be yes/no/uncertain")
    if result["event_scope"] not in EVENT_SCOPE:
        raise ValueError("invalid event_scope")
    if result["stance_quality"] not in STANCE_QUALITY:
        raise ValueError("invalid stance_quality")
    if result["route"] not in ROUTES:
        raise ValueError("invalid route")
    if not str(result["description"]).strip():
        raise ValueError("description is required")

    strict = (
        all(result[field] == "yes" for field in ATOMIC_YES)
        and result["event_scope"] in STRICT_SCOPE
        and result["stance_quality"] in STRICT_STANCE
        and all(
            str(result[field]).strip()
            for field in (
                "behavior_evidence_quote",
                "stance_evidence_quote",
                "normalized_behavior",
                "normalized_norm",
            )
        )
    )
    if (result["route"] == "strict_visual_search") != strict:
        raise ValueError("strict route does not match the atomic occurred-event contract")
    if result["route"] == "instructional_definition_retrieval":
        if result["event_scope"] != "generic_or_hypothetical":
            raise ValueError("instructional definition route requires generic/hypothetical scope")
        if result["occurred_event_grounded"] != "no":
            raise ValueError("instructional definition route cannot claim an occurred event")
    if result["route"] == "exploratory_contested" and result["stance_quality"] != "contested":
        raise ValueError("contested route requires contested stance evidence")
    if result["route"] == "exploratory_ambiguous" and not (
        result["event_scope"] in {"aggregate_examples", "ambiguous_or_unresolved"}
        or result["behavior_semantically_specific"] != "yes"
    ):
        raise ValueError("ambiguous route requires an explicit ambiguity")
    return {**result, "strict_event_candidate": strict, "automatic_acceptance": False}
