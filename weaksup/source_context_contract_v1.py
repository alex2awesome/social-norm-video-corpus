#!/usr/bin/env python3
"""Unified source-context labeling contract (YT + Dailymotion + Reddit).

One LLM pass per SOURCE over external context — title, uploader description,
tags, subreddit + post title + top comments (Reddit), and a transcript
excerpt — producing atomic fields.  The composite judgments
(``social_norm_apparent``, ``organic_capture_ok``) are derived in code, never
asked, per the standing atomic-decomposition rule.  This evidence is
detector-independent: the 2026-09-01 spot-check showed the detector
over-assigns social norms to fights, so external framing is the corrective
signal.  Shadow labels only.
"""

from __future__ import annotations

import json
from typing import Any

CONTRACT_VERSION = "source_context_v1"

CONTENT_TYPES = (
    "organic_capture", "staged_prank_or_experiment", "scripted_fiction_or_comedy",
    "news_or_documentary", "police_or_bodycam", "compilation",
    "vlog_or_commentary_only", "tutorial_or_lecture", "animation_or_game",
    "music_or_other", "unclear",
)
NORM_DOMAINS = (
    "interpersonal_social", "traffic_driving", "property", "authority_legal",
    "safety_physical", "violence_crime", "none_apparent", "unclear",
)
POLARITIES = ("violation", "correct_example", "both", "none", "unclear")
CROWD_STANCES = (
    "condemns_actor", "defends_actor", "split_or_debated", "spectacle_only",
    "no_comments", "unclear",
)
EVIDENCE_FIELDS = ("title", "description", "tags", "subreddit", "comments", "transcript")

SOCIAL_DOMAINS = {"interpersonal_social", "traffic_driving", "property"}

SYSTEM_PROMPT = """You label the SOURCE CONTEXT of a video for a research corpus \
about social norms. You see external framing only: platform, title, uploader \
description, tags, and for Reddit the subreddit, post title, and top comments, \
plus a transcript excerpt. You do NOT see the video.

Answer each atomic field from the evidence; use "unclear" or null when the \
evidence does not say. Do not guess. Comments and descriptions may be in any \
language. Titles like "prank", "social experiment", "performances", TV/episode \
naming, news chyron language, or gameplay references are strong content-type \
evidence. A concrete transgression named in the framing ("man cuts the whole \
queue", "attacked a senior and stole his wallet") supports norm fields; pure \
spectacle framing ("insane fight", "crazy knockout") supports \
violence_crime or none_apparent instead.

Return ONE JSON object, no prose, with exactly these keys:
{"content_type": one of %s,
 "norm_domain": one of %s,
 "inferred_behavior": short concrete actor-neutral behavior or null,
 "inferred_norm": short norm/expectation statement or null,
 "polarity": one of %s,
 "crowd_stance": one of %s (Reddit comments only; "no_comments" otherwise),
 "evidence_fields": subset of %s that actually supported your answers,
 "note": one short sentence}""" % (
    json.dumps(CONTENT_TYPES), json.dumps(NORM_DOMAINS), json.dumps(POLARITIES),
    json.dumps(CROWD_STANCES), json.dumps(EVIDENCE_FIELDS),
)


def packet_prompt(packet: dict[str, Any]) -> str:
    lines = [f"platform: {packet.get('platform')}"]
    for key in ("title", "uploader", "description", "tags"):
        value = packet.get(key)
        if value:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    if packet.get("subreddit"):
        lines.append(f"subreddit: r/{packet['subreddit']}"
                     + (f" [flair: {packet['flair']}]" if packet.get("flair") else ""))
    if packet.get("post_title"):
        lines.append(f"reddit post title: {json.dumps(packet['post_title'], ensure_ascii=False)}")
    for comment in packet.get("comments") or []:
        lines.append(f"comment (score {comment.get('score')}): "
                     + json.dumps(comment.get("body", ""), ensure_ascii=False))
    if packet.get("transcript_excerpt"):
        lines.append(f"transcript excerpt: {json.dumps(packet['transcript_excerpt'], ensure_ascii=False)}")
    for excerpt in packet.get("reaction_excerpts") or []:
        lines.append(f"detected reaction: {json.dumps(excerpt, ensure_ascii=False)}")
    return "\n".join(lines)


def validate_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("content_type") not in CONTENT_TYPES:
        raise ValueError(f"invalid content_type: {result.get('content_type')!r}")
    if result.get("norm_domain") not in NORM_DOMAINS:
        raise ValueError(f"invalid norm_domain: {result.get('norm_domain')!r}")
    if result.get("polarity") not in POLARITIES:
        raise ValueError(f"invalid polarity: {result.get('polarity')!r}")
    if result.get("crowd_stance") not in CROWD_STANCES:
        raise ValueError(f"invalid crowd_stance: {result.get('crowd_stance')!r}")
    fields = result.get("evidence_fields")
    if not isinstance(fields, list) or not set(fields) <= set(EVIDENCE_FIELDS):
        raise ValueError(f"invalid evidence_fields: {fields!r}")
    for key in ("inferred_behavior", "inferred_norm"):
        value = result.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be null or nonempty string")
    # Composites are derived, never model-supplied.
    derived = {
        "social_norm_apparent": (
            "yes" if result["norm_domain"] in SOCIAL_DOMAINS
            and result.get("inferred_behavior")
            else "no" if result["norm_domain"] in {"none_apparent", "violence_crime"}
            else "unclear"
        ),
        "organic_capture_ok": result["content_type"] == "organic_capture",
        "reroute_hint": (
            "instructional_review"
            if result["content_type"] == "staged_prank_or_experiment" else
            "commentary_review"
            if result["content_type"] in {"news_or_documentary", "vlog_or_commentary_only"}
            else None
        ),
        "contract_version": CONTRACT_VERSION,
        "acceptance_label": None,
    }
    return {**result, **derived}
