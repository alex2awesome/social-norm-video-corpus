#!/usr/bin/env python3
"""Build a shadow-only manifest of concretely title-labeled visual events."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


CAPTURE = r"(?:caught|captured|footage|video|cctv|dashcam|camera|on tape)"
EVENT = (
    r"(?:attack|assault|beat|beating|hit|strik|kick|punch|harass|steal|rob|"
    r"confront|berat|fight|brawl|road rage|reckless|dangerous driv|overtak|"
    r"bully|abuse|slap|shove|spit)"
)
ACTOR = (
    r"(?:driver|motorist|customer|passenger|man|woman|teacher|student|police|"
    r"cop|officer|guard|worker|waiter|bystander|mob|group|crowd|boss|employee)"
)
ACTION = (
    r"(?:attacks?|assaults?|beats?|hits?|strikes?|kicks?|punches?|harasses?|"
    r"steals?|robs?|confronts?|berates?|bullies?|threatens?|abuses?|slaps?|"
    r"shoves?|spits?)"
)

PATTERNS = {
    "capture_before_event": re.compile(
        rf"\b{CAPTURE}\b.{{0,70}}\b{EVENT}", re.IGNORECASE
    ),
    "event_before_capture": re.compile(
        rf"\b{EVENT}.{{0,70}}\b{CAPTURE}\b", re.IGNORECASE
    ),
    "actor_action": re.compile(
        rf"\b{ACTOR}\b.{{0,45}}\b{ACTION}\b", re.IGNORECASE
    ),
}
DISCUSSION_ONLY = re.compile(
    r"\b(discusses?|talks? about|interview|explains?|reacts? to|"
    r"speaks? about|accuses?|alleges?|claims?)\b",
    re.IGNORECASE,
)
ANIMAL = re.compile(
    r"\b(dog|cat|snake|shark|bear|monkey|ape|lion|tiger|leopard|animal)\b",
    re.IGNORECASE,
)
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}


def title_cues(title: str) -> list[str]:
    return [name for name, pattern in PATTERNS.items() if pattern.search(title)]


def iter_metadata(discussion_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in discussion_dir.glob("*.json"):
        try:
            source = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "uid": str(source.get("video_id") or path.stem),
                "title": str(source.get("title") or "").strip(),
                "agent": str(source.get("agent") or "unknown").lower(),
                "source": source.get("source"),
                "query_source": (source.get("provenance") or {}).get("query_source"),
                "found_by_query": (source.get("provenance") or {}).get(
                    "found_by_query"
                ),
            }
        )
    return rows


def find_media(media_dir: Path, uid: str) -> Path | None:
    matches = [
        path
        for path in media_dir.glob(f"{uid}.*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
        and path.stat().st_size > 0
    ]
    return matches[0] if len(matches) == 1 else None


def build_candidates(
    metadata_rows: list[dict[str, Any]], media_dir: Path
) -> list[dict[str, Any]]:
    candidates = []
    for source in metadata_rows:
        cues = title_cues(source["title"])
        if not cues or source["agent"] == "animal" or ANIMAL.search(source["title"]):
            continue
        media = find_media(media_dir, source["uid"])
        if media is None:
            continue
        candidates.append(
            {
                "item_id": f"commentary:{source['uid']}:0",
                "uid": source["uid"],
                "pillar": "commentary",
                "source_path": str(media.resolve()),
                "source_exists": True,
                # render_full_corpus_score_audit seals this field and never puts
                # it on the blind contact sheet.
                "norm": source["title"],
                "query_source": source["query_source"],
                "found_by_query": source["found_by_query"],
                "title_event_cues": cues,
                "discussion_only_risk": bool(DISCUSSION_ONLY.search(source["title"])),
                "agent": source["agent"],
            }
        )
    return candidates


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discussion-dir", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--require-cue",
        action="append",
        choices=tuple(PATTERNS),
        default=[],
        help="Keep only candidates containing every requested cue.",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)

    candidates = build_candidates(iter_metadata(args.discussion_dir), args.media_dir)
    if args.require_cue:
        required = set(args.require_cue)
        candidates = [
            row
            for row in candidates
            if required.issubset(row["title_event_cues"])
        ]
    manifest = args.out / "candidate_manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in candidates
        )
    )
    summary = {
        "kind": "commentary_title_labeled_event_shadow",
        "policy": "audit_prioritization_only_no_keep_reject",
        "required_cues": args.require_cue,
        "candidates": len(candidates),
        "discussion_only_risk": sum(
            row["discussion_only_risk"] for row in candidates
        ),
        "cue_counts": dict(
            Counter(cue for row in candidates for cue in row["title_event_cues"])
        ),
        "candidate_manifest_sha256": file_sha256(manifest),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
