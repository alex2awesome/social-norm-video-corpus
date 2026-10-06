#!/usr/bin/env python3
"""Score explicit production/staging cues for witnessed sources.

This is a shadow-only source-level reroute candidate.  It never deletes media
or changes pillar labels.  The purpose is to identify deliberately produced
pranks, social experiments, and creator setups that may still be useful as
instructional demonstrations but should not be assumed to be organic events.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


TITLE_STAGING = re.compile(
    r"\b(?:prank(?:ster|sters|ed|ing|s)?|social experiment|hidden camera)\b",
    re.IGNORECASE,
)
# Candidate only until a source-disjoint visual confirmation passes the same
# promotion gate as the established title cue.  These programs deliberately
# stage a scenario for (often genuine) bystander responses, making them useful
# instructional material but not organic witnessed footage.
TITLE_WWYD_CANDIDATE = re.compile(
    r"\bwhat\s+would\s+you\s+do\b|\bwwyd\b",
    re.IGNORECASE,
)
OFFICIAL_WWYD_CHANNEL = "what would you do?"
# Audited cue for creator-initiated encounters whose titles omit the words
# prank/experiment. It is kept separate from title_staging_cue for provenance.
TITLE_CREATOR_INITIATED = re.compile(
    r"\b(?:trying to|attempting to|how to)\s+"
    r"(?:kiss|hit on|pick up|touch|hug|scare|annoy|fight|propose to)\s+"
    r"(?:random\s+)?(?:girls|women|men|guys|strangers|people)\b"
    # Bare "touching women" is often a reporter's description of misconduct,
    # not evidence that the uploader initiated it. The explicit
    # "trying/attempting/how to touch ..." branch above remains eligible.
    r"|\b(?:kissing|hitting on|picking up|hugging|scaring|annoying)\s+"
    r"(?:random\s+)?(?:girls|women|men|guys|strangers|people)\b",
    re.IGNORECASE,
)
EXPLICIT_REVEAL = re.compile(
    r"\b(?:"
    r"it(?:'s| is)|this is|that(?:'s| is)|"
    r"we(?:'re| are)(?: just)?(?: doing)?|just doing"
    r") (?:a )?(?:joke|prank)\b"
    r"|\b(?:you(?:'re| are)|we(?:'re| are)) on camera\b"
    r"|\b(?:we are|we're) (?:filming|recording) (?:this|you)\b",
    re.IGNORECASE,
)
CREATOR_INTRO = re.compile(
    r"\b(?:what(?:'s| is) up|yo) (?:guys|everybody)\b"
    r"|\b(?:hello|hey) (?:guys|everybody|ladies and gentlemen)\b",
    re.IGNORECASE,
)
CREATOR_PLAN = re.compile(
    r"\btoday (?:i(?:'m| am)|we(?:'re| are)) (?:going to|gonna)\b"
    r"|\blet(?:'s| us) see (?:how|if|what)\b"
    r"|\bgoing (?:up|around) to (?:people|strangers)\b",
    re.IGNORECASE,
)
CREATOR_CTA = re.compile(
    r"\b(?:leave|put) (?:it|your idea|your ideas) in the comments\b"
    r"|\b(?:like and subscribe|subscribe to (?:my|the) channel)\b",
    re.IGNORECASE,
)


def transcript_text(payload: dict[str, Any]) -> str:
    segments = payload.get("segments") or []
    return " ".join(str(segment.get("text") or "") for segment in segments)


def matched_examples(pattern: re.Pattern[str], text: str, limit: int = 3) -> list[str]:
    return [match.group(0) for match in list(pattern.finditer(text))[:limit]]


def score_text(
    title: str | None,
    transcript: str,
    channel: str | None = None,
) -> dict[str, Any]:
    title = title or ""
    normalized_channel = (channel or "").strip().casefold()
    title_matches = matched_examples(TITLE_STAGING, title)
    wwyd_title_matches = matched_examples(TITLE_WWYD_CANDIDATE, title)
    official_wwyd_channel = normalized_channel == OFFICIAL_WWYD_CHANNEL
    initiated_title_matches = matched_examples(TITLE_CREATOR_INITIATED, title)
    reveal_matches = matched_examples(EXPLICIT_REVEAL, transcript)
    intro_matches = matched_examples(CREATOR_INTRO, transcript)
    plan_matches = matched_examples(CREATOR_PLAN, transcript)
    cta_matches = matched_examples(CREATOR_CTA, transcript)
    creator_setup = bool(intro_matches and (plan_matches or cta_matches))
    explicit = bool(
        title_matches or reveal_matches or creator_setup or official_wwyd_channel
    )
    audited_organic_exclusion = bool(
        title_matches or initiated_title_matches or official_wwyd_channel
    )
    # These names are the exact registry rule IDs.  Keeping them alongside the
    # diagnostic fields lets the fail-closed combiner consume this score file
    # without guessing which exploratory cues were actually promoted.
    audited_creator_title = bool(title_matches or initiated_title_matches)
    return {
        "title_staging_cue": bool(title_matches),
        "title_wwyd_candidate_cue": bool(wwyd_title_matches),
        "official_wwyd_channel_cue": official_wwyd_channel,
        "title_creator_initiated_candidate_cue": bool(
            initiated_title_matches
        ),
        "transcript_explicit_reveal_cue": bool(reveal_matches),
        "transcript_creator_setup_cue": creator_setup,
        "explicit_staging_candidate": explicit,
        "audited_strict_organic_exclusion_candidate": audited_organic_exclusion,
        "witnessed_creator_staging_title_v2": audited_creator_title,
        "witnessed_official_wwyd_channel_v1": official_wwyd_channel,
        "cue_count": sum(
            [
                bool(title_matches),
                bool(reveal_matches),
                creator_setup,
                bool(cta_matches),
            ]
        ),
        "matches": {
            "title": title_matches,
            "title_wwyd_candidate": wwyd_title_matches,
            "title_creator_initiated_candidate": initiated_title_matches,
            "explicit_reveal": reveal_matches,
            "creator_intro": intro_matches,
            "creator_plan": plan_matches,
            "creator_cta": cta_matches,
        },
        "policy": "shadow_reroute_candidate_preserve_original",
        "candidate_policy": (
            "audited_source_level_non_destructive_reroute_only; "
            "instructional_demo_requires_separate_visual_contract"
        ),
    }


def title_map(db_path: Path) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        return {
            str(row["video_id"]): {
                "title": row["title"],
                "channel": row["channel"],
                "query": row["query"],
                "query_source": row["query_source"],
            }
            for row in connection.execute(
                "SELECT video_id, title, channel, query, query_source FROM seen_videos"
            )
        }
    finally:
        connection.close()


def score_corpus(root: Path) -> list[dict[str, Any]]:
    titles = title_map(root / "data" / "state.db")
    rows = []
    for metadata_path in sorted((root / "data" / "hits").glob("*/metadata.json")):
        uid = metadata_path.parent.name
        transcript_path = root / "data" / "transcripts" / f"{uid}.json"
        if not transcript_path.is_file():
            rows.append(
                {
                    "uid": uid,
                    "metadata_path": str(metadata_path),
                    "transcript_path": str(transcript_path),
                    "error": "missing_transcript",
                    "policy": "shadow_reroute_candidate_preserve_original",
                }
            )
            continue
        try:
            transcript = transcript_text(json.loads(transcript_path.read_text()))
            title_record = titles.get(uid) or {}
            score = score_text(
                title_record.get("title"),
                transcript,
                title_record.get("channel"),
            )
            rows.append(
                {
                    "uid": uid,
                    "metadata_path": str(metadata_path),
                    "transcript_path": str(transcript_path),
                    "title": title_record.get("title"),
                    "channel": title_record.get("channel"),
                    "query": title_record.get("query"),
                    "query_source": title_record.get("query_source"),
                    **score,
                    "error": None,
                }
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "uid": uid,
                    "metadata_path": str(metadata_path),
                    "transcript_path": str(transcript_path),
                    "error": f"{type(exc).__name__}: {exc}",
                    "policy": "shadow_reroute_candidate_preserve_original",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    rows = score_corpus(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    )
    summary = {
        "items": len(rows),
        "successful": sum(row.get("error") is None for row in rows),
        "failed": sum(row.get("error") is not None for row in rows),
        "explicit_staging_candidates": sum(
            row.get("explicit_staging_candidate") is True for row in rows
        ),
        "official_wwyd_channel_cues": sum(
            row.get("official_wwyd_channel_cue") is True for row in rows
        ),
        "broad_wwyd_title_candidates": sum(
            row.get("title_wwyd_candidate_cue") is True for row in rows
        ),
        "policy": "shadow_reroute_candidate_preserve_original",
        "out": str(args.out),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
