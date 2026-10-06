#!/usr/bin/env python3
"""Lightweight atomic contract for strict witnessed-reaction routing."""

from __future__ import annotations

from typing import Any


SOCIAL_TRIGGERS = {"interpersonal_treatment", "shared_public_conduct"}
TARGETED_RESPONSE_CONTENT = {
    "targeted_objection",
    "correction_or_sanction",
    "protective_intervention",
    "interposition_or_separation",
}


def candidate_positive(
    result: dict[str, Any], *, include_authority: bool, require_social: bool,
    require_unstaged: bool = True,
) -> bool:
    """Require a grounded, targeted, temporally linked organic response.

    ``clearly_staged`` examples remain useful enacted demonstrations, but they
    are not organic witnessed evidence.  Missing or uncertain staging fails
    closed here; a more permissive rerouter may preserve those clips for
    instructional review.
    """
    roles = {"separate_bystander", "organic_audience"}
    if include_authority:
        roles.add("authority_or_host")
    return (
        result.get("reaction_grounded") == "yes"
        and result.get("action_before_or_overlaps_response") == "yes"
        and result.get("response_targets_action") == "yes"
        and result.get("responder_role") in roles
        and result.get("response_content") in TARGETED_RESPONSE_CONTENT
        and (
            not require_unstaged
            or result.get("staging") == "no_clear_staging_evidence"
        )
        and (
            not require_social
            or result.get("trigger_kind") in SOCIAL_TRIGGERS
        )
    )
