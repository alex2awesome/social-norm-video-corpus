#!/usr/bin/env python3
"""Score a transparent utterance anchor for dialogue-only visual demos.

This is a shadow feature. It does not mutate corpus state. Physical and
explicitly depicted actions pass unchanged. A candidate whose V9A evidence
source is ``situated_dialogue_or_subtitles`` passes only when the V9A event
evidence contains a quoted utterance with at least two word tokens.
"""

from __future__ import annotations

import re
from typing import Any


EVIDENCE_FIELDS = (
    "literal_action_or_situated_utterance",
    "evidence_before",
    "evidence_during",
    "evidence_after",
    "evidence",
)

_QUOTED_SEGMENTS = (
    re.compile(r'"([^"”]{2,})["”]'),
    re.compile(r"“([^”]{2,})”"),
    # The closing delimiter must not be followed by a word character. This
    # lets an apostrophe inside a contraction remain part of the quoted span.
    re.compile(r"(?<!\w)'(.{2,}?)'(?!\w)"),
    re.compile(r"(?<!\w)‘(.{2,}?)’(?!\w)"),
)
_WORD_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def quoted_utterances(text: str) -> list[str]:
    """Return nontrivial quoted spans without treating apostrophes as quotes."""

    spans: list[str] = []
    for pattern in _QUOTED_SEGMENTS:
        for match in pattern.finditer(text):
            span = match.group(1).strip()
            if len(_WORD_TOKEN.findall(span)) >= 2 and span not in spans:
                spans.append(span)
    return spans


def utterance_anchor(result_or_record: dict[str, Any]) -> dict[str, Any]:
    """Return the deterministic V15 anchor decision and its evidence.

    ``result_or_record`` may be a V9A result directly or a complete V9A record
    containing a ``result`` mapping.
    """

    result = result_or_record.get("result", result_or_record)
    if not isinstance(result, dict):
        raise ValueError("V9A result must be a mapping")

    evidence_source = str(result.get("evidence_source", "")).strip()
    if evidence_source != "situated_dialogue_or_subtitles":
        return {
            "passes": True,
            "reason": "non_dialogue_visual_action",
            "evidence_field": None,
            "quoted_utterance": None,
        }

    for field in EVIDENCE_FIELDS:
        text = str(result.get(field, ""))
        spans = quoted_utterances(text)
        if spans:
            return {
                "passes": True,
                "reason": "specific_quoted_utterance",
                "evidence_field": field,
                "quoted_utterance": spans[0],
            }
    return {
        "passes": False,
        "reason": "dialogue_without_specific_utterance",
        "evidence_field": None,
        "quoted_utterance": None,
    }


def passes_utterance_anchor(result_or_record: dict[str, Any]) -> bool:
    return bool(utterance_anchor(result_or_record)["passes"])
