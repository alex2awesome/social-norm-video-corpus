#!/usr/bin/env python3
"""Categorize detector-assigned norm strings (append-only tagging pass).

The live detector stores a freeform ``norm`` per reaction.  The 2026-09-01
spot-check found the dominant witnessed failure is ``no_norm``: reactions to
fights, danger, or generic upset rather than a social-norm violation — and
the freeform strings encode exactly this (affect labels like "surprise",
safety labels like "public safety", violence labels like "aggression").
This module maps norm strings into a small category set so dataset views can
be filtered and a future LF can vote.  Tags only; nothing is relabeled.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

TAGGER_VERSION = "norm_category_tags_v1"

# Order matters: first match wins.  Patterns run on the lowercased string.
CATEGORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("affect_only", re.compile(
        r"^(?:surprise|shock|concern|confusion|frustration|anger|fear|"
        r"disbelief|amusement|excitement|annoyance|distress|panic|outrage)$")),
    ("traffic_driving", re.compile(
        r"driv|traffic|road|pedestrian|crosswalk|parking|lane|speed|"
        r"right of way|tailgat|merge|vehicle")),
    ("violence_conflict", re.compile(
        r"aggress|violence|violen|assault|fight|threat|attack|abuse|"
        r"intimidat|harm|weapon|hitting|self.?defen[cs]e")),
    ("safety_physical", re.compile(
        r"safety|danger|caution|hazard|injur|protect|emergency|fire|"
        r"supervis")),
    # Interpersonal outranks authority/property so "respect for authority"
    # and "respect for property" resolve as respect-norms.
    ("interpersonal_social", re.compile(
        r"respect|polite|courtes|manner|civil|rude|kind|empath|fair|"
        r"honest|accountab|apolog|consent|privacy|personal space|boundar|"
        r"consideration|patien|queue|line|turn|greet|hospitalit|gratitude|"
        r"sharing|professional|decen|dignit|toleran|inclusion|"
        r"harass|discriminat|bully|mock|humiliat|gossip|"
        r"interrupt|noise|quiet|clean|hygien|space|help|care|support|"
        r"lying|cheat|loyal|trust|fidelity|etiquette|modest|decorum|"
        r"public behavior|social")),
    ("authority_legal", re.compile(
        r"authorit|police|law|legal|compliance|arrest|court|rules?$|"
        r"regulation|obey|order")),
    ("property", re.compile(
        r"property|theft|steal|vandal|damage|litter|belongings|borrow")),
)

# Categories that name a plausible social-norm domain (a future LF would
# vote +1 on norm_event_supported); affect_only would vote -1; the rest are
# neutral tags for filtering, not votes.
SOCIAL_CATEGORIES = {"interpersonal_social", "traffic_driving", "property"}


def categorize_norm(norm: Any) -> str:
    text = " ".join(str(norm or "").lower().split())
    if not text:
        return "missing"
    for category, pattern in CATEGORY_PATTERNS.items() if isinstance(
        CATEGORY_PATTERNS, dict
    ) else CATEGORY_PATTERNS:
        if pattern.search(text):
            return category
    return "other_unmapped"


def categorize_item(reactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Item-level roll-up: every reaction's norm categorized; the item's
    primary category is the most social one present (a single genuine social
    norm among affect labels still makes the item social)."""
    categories = [categorize_norm(r.get("norm")) for r in reactions]
    counted = Counter(categories)
    priority = (
        "interpersonal_social", "property", "traffic_driving",
        "authority_legal", "safety_physical", "violence_conflict",
        "affect_only", "other_unmapped", "missing",
    )
    primary = next((c for c in priority if c in counted), "missing")
    return {
        "primary_category": primary,
        "categories": dict(counted),
        "any_social_norm_category": bool(set(counted) & SOCIAL_CATEGORIES),
        "affect_only_item": set(counted) <= {"affect_only", "missing"} and bool(counted),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, Any]] = {}
    counts: Counter = Counter()
    with args.out.open("x") as out, args.proposals.open() as proposals:
        for line in proposals:
            if not line.strip():
                continue
            row = json.loads(line)
            uid, clip_idx = row["uid"], row["clip_idx"]
            if uid not in cache:
                try:
                    cache[uid] = json.loads(
                        (args.root / "data" / "hits" / uid / "metadata.json").read_text()
                    )
                except (OSError, json.JSONDecodeError):
                    cache[uid] = {}
            reactions = [
                r for r in cache[uid].get("reactions") or []
                if str(r.get("clip_idx")) == str(clip_idx)
            ]
            tag = categorize_item(reactions)
            counts[tag["primary_category"]] += 1
            out.write(json.dumps({
                "item_id": row["item_id"], "uid": uid, "clip_idx": clip_idx,
                "evidence_tier": row.get("evidence_tier"), **tag,
                "tagger_version": TAGGER_VERSION,
            }, sort_keys=True) + "\n")
    print(json.dumps({"tagger_version": TAGGER_VERSION,
                      "primary_category_counts": dict(counts.most_common())},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
