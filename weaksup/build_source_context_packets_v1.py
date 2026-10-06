#!/usr/bin/env python3
"""Build unified per-source context packets for the LLM labeling pass.

Joins, per source uid across YT/Dailymotion/Reddit/Rumble: harvested sidecar
metadata (title/description/tags/uploader), Reddit post + top comments, and a
transcript excerpt with detected-reaction snippets for witnessed sources.
Append-only packets; the GPU runner consumes them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

BUILDER_VERSION = "build_source_context_packets_v1"

TRANSCRIPT_EXCERPT_CHARS = 1200
DESCRIPTION_CHARS = 1500
MAX_COMMENTS = 10
COMMENT_CHARS = 220
MAX_TAGS = 15


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def transcript_context(root: Path, uid: str, reactions: list[dict[str, Any]]) -> tuple[str, list[str]]:
    path = root / "data" / "transcripts" / f"{uid}.json"
    if not path.is_file():
        return "", []
    try:
        segments = json.loads(path.read_text()).get("segments") or []
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "", []
    excerpt = " ".join(str(s.get("text") or "") for s in segments)[:TRANSCRIPT_EXCERPT_CHARS]
    snippets = []
    for reaction in reactions[:3]:
        text = str(reaction.get("matched_text") or reaction.get("phrase") or "").strip()
        if text:
            snippets.append(text[:200])
    return excerpt.strip(), snippets


def build_packet(
    uid: str,
    pillars: list[str],
    metadata: dict[str, Any] | None,
    reddit: dict[str, Any] | None,
    root: Path,
    reactions: list[dict[str, Any]],
) -> dict[str, Any]:
    excerpt, reaction_excerpts = transcript_context(root, uid, reactions)
    packet: dict[str, Any] = {
        "uid": uid, "platform": uid.split("__")[0], "pillars": sorted(set(pillars)),
        "builder_version": BUILDER_VERSION,
    }
    if metadata:
        for key in ("title", "uploader", "channel"):
            if metadata.get(key):
                packet[key] = metadata[key]
        if metadata.get("description"):
            packet["description"] = str(metadata["description"])[:DESCRIPTION_CHARS]
        if metadata.get("tags"):
            packet["tags"] = list(metadata["tags"])[:MAX_TAGS]
    if reddit:
        post = reddit.get("post") or {}
        packet["subreddit"] = post.get("subreddit")
        packet["post_title"] = post.get("title")
        packet["flair"] = post.get("link_flair_text")
        packet["comments"] = [
            {"body": str(c.get("body") or "")[:COMMENT_CHARS], "score": c.get("score")}
            for c in (reddit.get("comments") or [])[:MAX_COMMENTS]
        ]
        packet.setdefault("title", post.get("title"))
    if excerpt:
        packet["transcript_excerpt"] = excerpt
    if reaction_excerpts:
        packet["reaction_excerpts"] = reaction_excerpts
    packet["context_available"] = sorted(
        k for k in ("title", "description", "tags", "subreddit", "comments",
                    "transcript_excerpt") if packet.get(k)
    )
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reddit-context", type=Path, default=None)
    parser.add_argument("--uid-pillars", type=Path, required=True,
                        help="JSONL of {uid, pillar} rows defining scope")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {r["uid"]: r for r in iter_jsonl(args.metadata)}
    reddit = {}
    if args.reddit_context and args.reddit_context.is_file():
        reddit = {r["uid"]: r for r in iter_jsonl(args.reddit_context)}
    scope: dict[str, list[str]] = {}
    for row in iter_jsonl(args.uid_pillars):
        scope.setdefault(row["uid"], []).append(row["pillar"])
    counts = {"packets": 0, "with_metadata": 0, "with_reddit": 0, "with_transcript": 0}
    with args.out.open("x") as out:
        for uid, pillars in sorted(scope.items()):
            reactions = []
            if "witnessed" in pillars:
                meta_path = args.root / "data" / "hits" / uid / "metadata.json"
                if meta_path.is_file():
                    try:
                        reactions = json.loads(meta_path.read_text()).get("reactions") or []
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
            packet = build_packet(uid, pillars, metadata.get(uid), reddit.get(uid),
                                  args.root, reactions)
            counts["packets"] += 1
            counts["with_metadata"] += uid in metadata
            counts["with_reddit"] += uid in reddit
            counts["with_transcript"] += "transcript_excerpt" in packet
            out.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"builder_version": BUILDER_VERSION, **counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
