"""Versioned, auditable scope gate for automatically generated queries.

This gate is deliberately narrower than the corpus definition.  It governs
only self-propagating query sources for the witnessed-reaction crawler; manual,
taxonomy, instructional, commentary, and negative-control queries are not
altered.  A rejection changes queue eligibility, never media or labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Optional


POLICY_VERSION = "ordinary_social_witnessed_v1"
AUTO_SOURCES = frozenset({"llm_expand", "exploit", "explore"})


@dataclass(frozen=True)
class ScopeDecision:
    allowed: bool
    reason: str
    policy_version: str = POLICY_VERSION
    matched_text: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


# Families below came from false-positive trajectories observed in the live
# 2026-08-07 recovery batch and earlier source-disjoint audits.  Patterns are
# phrase-level so ordinary human obligations involving an animal (for example,
# an owner refusing to clean up after a dog) are not rejected merely for saying
# "dog".
_BLOCK_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("authority_or_police", (
        r"\bpolice\b", r"\bcops?\b", r"\blaw enforcement\b", r"\bsheriffs?\b",
        r"\btroopers?\b", r"\bbody\s*cam\b", r"\btraffic stop\b",
        r"\b(officer|deputy)\s+(arrests?|detains?|shoots?|rescues?|confronts?)\b",
        r"\b(arrested|arrest|detained)\b",
    )),
    ("weapon_or_mass_violence", (
        r"\bmass shooting\b", r"\bschool shooting\b", r"\bactive shooter\b",
        r"\bgunman\b", r"\bshootout\b", r"\bshooting caught\b",
        r"\b(stabbing|knife attack|terrorist attack|hostage)\b",
    )),
    ("accident_or_disaster_spectacle", (
        r"\b(car|bus|truck|train|plane|motorcycle)\s+(accident|crash|wreck)\b",
        r"\b(accident|crash|wreck)\s+(caught|compilation|footage|video)\b",
        r"\b(crash|accident) compilation\b", r"\bnatural disaster\b",
        r"\b(explosion|building collapse)\s+(caught|footage|video)\b",
    )),
    ("nonhuman_spectacle", (
        r"\banimal fights?\b", r"\b(cat|dog|snake|bear|shark|lion|tiger)\s+vs\b",
        r"\bvs\s+(cat|dog|snake|bear|shark|lion|tiger)\b",
        r"\b(pet|wild|zoo)?\s*(animal|snake|bear|shark|lion|tiger)\s+attacks?\b",
        r"\b(cat|dog|pet)\s+(attacks?|bites?|fights?)\s+(cat|dog|person|owner)\b",
    )),
    ("gameplay_or_virtual", (
        r"\bgame\s*play\b", r"\bfortnite\b", r"\bgta\b", r"\broblox\b",
        r"\bminecraft\b", r"\bvideo game\b", r"\bstreamer highlights?\b",
    )),
    ("sports_or_combat_entertainment", (
        r"\bwwe\b", r"\baew\b", r"\bwrestl", r"\bsmackdown\b", r"\bcm punk\b",
        r"\bboxing (match|fight|knockout)\b", r"\b(mma|ufc) (fight|knockout)\b",
    )),
    ("produced_or_adult_spectacle", (
        r"\bdhar\s*mann\b", r"\bsocial experiment\b", r"\bprank compilation\b",
        r"\b(exhibitionis|nude|naked|nsfw|porn|onlyfans)\w*\b",
    )),
    ("technical_or_business_topic", (
        r"\b(search engine optimization|digital marketing|site backup)\b",
        r"\b(nfc|dim[_ -]?weight)\b", r"\bsoftware tutorial\b",
    )),
)

_COMPILED = tuple(
    (family, tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns))
    for family, patterns in _BLOCK_FAMILIES
)


def audit_query_scope(query: str, source: str) -> ScopeDecision:
    """Return the scope decision for one proposal without changing state."""
    if source not in AUTO_SOURCES:
        return ScopeDecision(True, "not_automatic_expansion")
    normalized = " ".join((query or "").split())
    if not normalized:
        return ScopeDecision(False, "empty_query")
    if len(normalized) > 120:
        return ScopeDecision(False, "query_too_long")
    for family, patterns in _COMPILED:
        for pattern in patterns:
            match = pattern.search(normalized)
            if match:
                return ScopeDecision(False, family, matched_text=match.group(0))
    return ScopeDecision(True, "within_ordinary_social_scope")

