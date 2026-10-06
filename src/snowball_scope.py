"""Audited parent gate for Dailymotion related-video snowball expansion.

The decision controls only whether a witnessed hit may create/follow a related
query.  It never changes the hit, its clips, or its labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Optional

from .query_scope import audit_query_scope


POLICY_VERSION = "witnessed_parent_v1"


@dataclass(frozen=True)
class SnowballDecision:
    allowed: bool
    reason: str
    policy_version: str = POLICY_VERSION
    matched_text: Optional[str] = None
    evidence: Optional[dict[str, Any]] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_TITLE_BLOCKS: tuple[tuple[str, re.Pattern], ...] = tuple(
    (reason, re.compile(pattern, re.IGNORECASE))
    for reason, pattern in (
        ("road_accident_or_rage", r"\b(road rage|accidents?|crash(?:es)?|wreck(?:s)?|car chase|vehicle pursuit|dash\s*cam)\b"),
        ("produced_or_compilation", r"\b(pranks?|compilation|movie|scripted|drama|episode|teleserye|skit)\b"),
        ("gameplay", r"\b(game\s*play|fortnite|gta|roblox|minecraft|save the world pve)\b"),
        ("police_or_arrest", r"\b(police|cops?|officers?|sheriffs?|body\s*cam|arrest(?:ed|ing|s)?|drug dealer)\b"),
        ("protest_or_spectacle", r"\b(mass strike|protests?|riot(?:s|ing)?)\b"),
        ("sexualized_prank", r"\b(sex prank|foursome|legs open)\b"),
    )
)


def audit_snowball_parent(
    *, title: str, agent: Optional[str], scene: Optional[dict],
    reaction_count: int,
) -> SnowballDecision:
    """Fail closed unless parent metadata describes a strong bystander scene.

    These are propagation criteria, not clip-acceptance criteria.  A useful
    clip can still be a poor graph seed when its title/neighborhood is dominated
    by police, crashes, pranks, gameplay, or other spectacle.
    """
    title = " ".join((title or "").split())
    scene = scene or {}
    evidence = {
        "agent": agent,
        "scene_type": scene.get("scene_type"),
        "n_people": scene.get("n_people"),
        "reactor_role": scene.get("reactor_role"),
        "reaction_strength": scene.get("reaction_strength"),
        "reaction_count": reaction_count,
    }
    generic_scope = audit_query_scope(title, "llm_expand")
    if not generic_scope.allowed:
        return SnowballDecision(
            False, f"title_scope:{generic_scope.reason}",
            matched_text=generic_scope.matched_text, evidence=evidence)
    for reason, pattern in _TITLE_BLOCKS:
        match = pattern.search(title)
        if match:
            return SnowballDecision(False, reason, matched_text=match.group(0), evidence=evidence)
    if agent not in {None, "human"}:
        return SnowballDecision(False, "nonhuman_agent", evidence=evidence)
    if scene.get("scene_type") != "action":
        return SnowballDecision(False, "not_live_action", evidence=evidence)
    try:
        n_people = int(scene.get("n_people") or 0)
        strength = int(scene.get("reaction_strength") or 0)
    except (TypeError, ValueError):
        return SnowballDecision(False, "invalid_scene_metadata", evidence=evidence)
    if n_people < 3:
        return SnowballDecision(False, "fewer_than_three_people", evidence=evidence)
    if scene.get("reactor_role") != "bystander":
        return SnowballDecision(False, "reactor_not_distinct_bystander", evidence=evidence)
    if strength < 3:
        return SnowballDecision(False, "weak_or_uncertain_reaction", evidence=evidence)
    if reaction_count < 1:
        return SnowballDecision(False, "no_saved_reaction", evidence=evidence)
    return SnowballDecision(True, "strong_in_scope_bystander_parent", evidence=evidence)

