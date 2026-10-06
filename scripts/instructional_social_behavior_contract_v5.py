#!/usr/bin/env python3
"""Fail-closed text-side social-behavior contract for instructional demos.

This shadow contract separates "the clip demonstrates something" from "the
claimed thing is a social norm."  It was motivated by complete manual review of
the V23 false positives, where both VLMs correctly saw a procedure but wrongly
promoted self-defense, tire changing, parking, clay work, generic self-help,
and analogy graphics as social-norm demonstrations.
"""

from __future__ import annotations

from typing import Any


TRI = {"yes", "no", "uncertain"}
DOMAINS = {
    "interpersonal_treatment",
    "communicative_act",
    "role_or_care_obligation",
    "shared_space_convention",
    "solo_social_convention",
    "procedural_skill",
    "personal_optimization",
    "object_or_product_procedure",
    "abstract_analogy",
    "technical_or_other",
    "unclear",
}
SOCIAL_DOMAINS = {
    "interpersonal_treatment",
    "communicative_act",
    "role_or_care_obligation",
    "shared_space_convention",
    "solo_social_convention",
}
ROUTES = {
    "strict_social_alignment",
    "relabel_required",
    "instructional_retrieval_only",
    "reject",
}
REQUIRED = {
    "social_behavior_domain",
    "specific_actor_role_grounded",
    "specific_behavior_grounded",
    "affected_person_or_shared_social_context_grounded",
    "normative_obligation_grounded",
    "literal_social_behavior_not_analogy",
    "procedural_skill_only",
    "personal_optimization_only",
    "proposed_norm_supported",
    "proposed_polarity_supported",
    "normalized_behavior",
    "normalized_norm",
    "behavior_evidence_quote",
    "norm_evidence_quote",
    "route",
    "description",
}


SYSTEM_PROMPT = """Audit an instructional detector's transcript label without
seeing title, category, query, or pixels. Decide whether the quoted material
grounds a specific social behavior: how people treat, communicate with, care
for, or coordinate with other people, or a recognizable convention in shared
social life.

Do not treat every demonstrated skill as a social norm. Fail procedural
competence, vehicle mechanics, self-defense technique, exercise, software,
cooking, tools, product handling, personal productivity, and internal self-care
when the claimed obligation is only to perform the task effectively. Fail an
analogy that merely stands for a social act: the eventual pixels must depict the
literal social behavior, not just tea, graphs, props, or metaphors.

A codified rule is not automatically excluded when the conduct remains a
socially legible action affecting people in a shared setting. Environmental or
safety conduct qualifies only when the quote grounds the affected people/shared
context and a social obligation, not just technique.

Use relabel_required when actor, behavior, affected person/shared context, and
obligation are fully grounded but the proposed norm is vague or wrong. Generic
social advice may remain instructional_retrieval_only, but it is not a strict
label and can never certify that a demo is visible. Return atomic fields and
verbatim quote evidence; do not infer missing roles or actions."""


def validate_result(result: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED - set(result)
    if missing:
        raise ValueError(f"missing instructional social V5 fields: {sorted(missing)}")
    if result["social_behavior_domain"] not in DOMAINS:
        raise ValueError("invalid social_behavior_domain")
    for field in (
        "specific_actor_role_grounded",
        "specific_behavior_grounded",
        "affected_person_or_shared_social_context_grounded",
        "normative_obligation_grounded",
        "literal_social_behavior_not_analogy",
        "procedural_skill_only",
        "personal_optimization_only",
        "proposed_norm_supported",
        "proposed_polarity_supported",
    ):
        if result[field] not in TRI:
            raise ValueError(f"{field} must be yes/no/uncertain")
    if result["route"] not in ROUTES:
        raise ValueError("invalid route")
    if not str(result["description"]).strip():
        raise ValueError("description is required")

    atomic_social = (
        result["social_behavior_domain"] in SOCIAL_DOMAINS
        and result["specific_actor_role_grounded"] == "yes"
        and result["specific_behavior_grounded"] == "yes"
        and result["affected_person_or_shared_social_context_grounded"] == "yes"
        and result["normative_obligation_grounded"] == "yes"
        and result["literal_social_behavior_not_analogy"] == "yes"
        and result["procedural_skill_only"] == "no"
        and result["personal_optimization_only"] == "no"
        and all(
            str(result[field]).strip()
            for field in ("normalized_behavior", "normalized_norm", "behavior_evidence_quote", "norm_evidence_quote")
        )
    )
    strict = (
        atomic_social
        and result["proposed_norm_supported"] == "yes"
        and result["proposed_polarity_supported"] == "yes"
    )
    relabel = atomic_social and (
        result["proposed_norm_supported"] in {"no", "uncertain"}
        or result["proposed_polarity_supported"] in {"no", "uncertain"}
    )
    if (result["route"] == "strict_social_alignment") != strict:
        raise ValueError("strict route does not match atomic social-label contract")
    if (result["route"] == "relabel_required") != relabel:
        raise ValueError("relabel route does not match atomic social-label contract")
    if result["route"] == "instructional_retrieval_only" and atomic_social:
        raise ValueError("fully grounded social behavior must use strict or relabel route")
    return {
        **result,
        "strict_text_candidate": strict,
        "text_candidate_after_relabel": relabel,
        "automatic_acceptance": False,
        "visual_demo_certified": False,
    }
