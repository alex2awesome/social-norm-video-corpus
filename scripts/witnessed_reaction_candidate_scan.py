#!/usr/bin/env python3
"""Scan an entire transcript for possible bystander-reaction windows.

This module is intentionally a high-recall proposal stage.  It does not assign
the reactor's identity or accept a witnessed datapoint.  Proposed windows must
be reranked by a role-aware video model or manual audit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

if __package__:
    from scripts.witnessed_reaction_text_features import intervention_features
else:
    from witnessed_reaction_text_features import intervention_features


TOKEN = re.compile(r"[a-z']+")


def _overlap(left_text: str, right_text: str) -> float:
    left = set(TOKEN.findall(left_text.lower()))
    right = set(TOKEN.findall(right_text.lower()))
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


@dataclass(frozen=True)
class ReactionCandidate:
    segment_index: int
    start: float
    end: float
    window_start: float
    window_end: float
    text: str
    mechanisms: tuple[str, ...]
    selected_quote_overlap: float
    negative_self_defense: bool
    negative_reported: bool
    generic_affect_only: bool


ACTIVE_MECHANISMS = (
    "direct",
    "distract",
    "delegate",
    "support",
    "evaluation",
    "warning",
)


def scan_reaction_candidates(
    segments: list[dict[str, Any]],
    *,
    selected_quote: str = "",
    pre_context_sec: float = 5.0,
    post_context_sec: float = 3.0,
    include_generic_affect: bool = False,
) -> list[ReactionCandidate]:
    """Return chronologically ordered high-recall candidate windows."""
    candidates = []
    for index, segment in enumerate(segments):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        features = intervention_features(text)
        mechanisms = tuple(
            name
            for name in ACTIVE_MECHANISMS
            if features[f"intervention.{name}"] > 0
        )
        generic = bool(features["intervention.generic_affect_only"])
        if not mechanisms and not (include_generic_affect and generic):
            continue
        start = float(
            segment.get("clip_start")
            if segment.get("clip_start") is not None
            else segment.get("start") or 0
        )
        end = float(
            segment.get("clip_end")
            if segment.get("clip_end") is not None
            else segment.get("end") or start
        )
        candidates.append(
            ReactionCandidate(
                segment_index=index,
                start=start,
                end=end,
                window_start=max(0.0, start - pre_context_sec),
                window_end=max(end, end + post_context_sec),
                text=text,
                mechanisms=mechanisms,
                selected_quote_overlap=_overlap(selected_quote, text),
                negative_self_defense=bool(features["intervention.self_defense"]),
                negative_reported=bool(features["intervention.reported"]),
                generic_affect_only=generic,
            )
        )
    return candidates


def candidate_scan_features(
    candidates: list[ReactionCandidate],
    *,
    selected_boundary: float | None = None,
) -> dict[str, float]:
    active = [
        candidate
        for candidate in candidates
        if not candidate.negative_self_defense and not candidate.negative_reported
    ]
    mechanism_counts = {
        name: sum(name in candidate.mechanisms for candidate in active)
        for name in ACTIVE_MECHANISMS
    }
    return {
        "candidate_scan.total": float(len(candidates)),
        "candidate_scan.active_nonnegative": float(len(active)),
        "candidate_scan.before_selected": float(
            sum(
                selected_boundary is not None and c.start < selected_boundary
                for c in active
            )
        ),
        "candidate_scan.after_selected": float(
            sum(
                selected_boundary is not None and c.start > selected_boundary
                for c in active
            )
        ),
        "candidate_scan.selected_quote_max_overlap": float(
            max((c.selected_quote_overlap for c in candidates), default=0.0)
        ),
        "candidate_scan.direct": float(mechanism_counts["direct"]),
        "candidate_scan.distract": float(mechanism_counts["distract"]),
        "candidate_scan.delegate": float(mechanism_counts["delegate"]),
        "candidate_scan.support": float(mechanism_counts["support"]),
        "candidate_scan.evaluation": float(mechanism_counts["evaluation"]),
        "candidate_scan.warning": float(mechanism_counts["warning"]),
    }


def serialize_candidates(
    candidates: list[ReactionCandidate],
) -> list[dict[str, Any]]:
    return [asdict(candidate) for candidate in candidates]
