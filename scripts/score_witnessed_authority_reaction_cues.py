#!/usr/bin/env python3
"""Score authority/enforcement reaction cues across witnessed metadata.

This is an append-only shadow scorer.  A positive cue means the reaction text
looks like a police/security/host command and therefore cannot independently
certify the strict organic-witnessed signal.  It does not reject or mutate the
source clip; visually useful matches remain scene candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


AUTHORITY_TAGS = {
    "arrest",
    "authority",
    "detention",
    "police order",
}

AUTHORITY_NORMS = {
    "authority assertion",
    "compliance with authority",
    "compliance with police orders",
    "law enforcement",
    "law enforcement action",
    "law enforcement involvement",
    "obey police orders",
    "respect law enforcement",
    "surrender",
}

AUTHORITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "under_or_threatened_arrest",
        re.compile(
            r"\b(?:(?:you(?:'re| are)|you'?re)\s+under\s+arrest|"
            r"(?:i(?:'m| am)|we(?:'re| are))\s+going\s+to\s+arrest\s+you)\b",
            re.I,
        ),
    ),
    (
        "jail_or_citation",
        re.compile(
            r"\b(?:(?:you(?:'re| are)|he(?:'s| is)|she(?:'s| is)|"
            r"they(?:'re| are))\s+going\s+to\s+(?:go\s+to\s+)?jail|"
            r"(?:i(?:'m| am)|we(?:'re| are))\s+(?:going\s+to\s+)?"
            r"send(?:ing)?\s+you\s+to\s+jail|"
            r"receive\s+(?:a\s+)?citation|issue\s+(?:you\s+)?(?:a\s+)?citation)\b",
            re.I,
        ),
    ),
    ("stay_where_you_are", re.compile(r"\bstay\s+where\s+you\s+are\b", re.I)),
    ("give_yourself_up", re.compile(r"\bgive\s+yourself\s+up\b", re.I)),
    (
        "get_on_floor_or_ground",
        re.compile(
            r"\b(?:get|lay|lie|stay)\s+(?:down\s+)?(?:on\s+)?the\s+"
            r"(?:floor|ground)\b",
            re.I,
        ),
    ),
    (
        "hands_command",
        re.compile(
            r"\b(?:show\s+(?:me\s+)?your\s+hands|"
            r"(?:put|get|keep)\s+your\s+hands?\s+"
            r"(?:behind|out(?:\s+of\s+your\s+pockets?)?|"
            r"where\s+i\s+can\s+see))\b",
            re.I,
        ),
    ),
    ("stop_resisting", re.compile(r"\bstop\s+resisting\b", re.I)),
    ("unlock_command", re.compile(r"\bunlock\s+(?:the\s+)?(?:door|window|car|vehicle|it)\b", re.I)),
    (
        "drop_weapon",
        re.compile(r"\b(?:drop|put\s+down)\s+(?:the\s+)?(?:gun|weapon|knife)\b", re.I),
    ),
    ("breath_test", re.compile(r"\b(?:breath|breathalyser|breathalyzer|drugs?)\s+test\b", re.I)),
    ("failed_to_stop", re.compile(r"\b(?:fail(?:ed)?|refus(?:e|ed))\s+to\s+stop\b", re.I)),
)


def normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def score_reaction(reaction: dict[str, Any]) -> dict[str, Any]:
    """Return enumerable authority cues for one reaction record."""
    tag = normalize(reaction.get("tag"))
    norm = normalize(reaction.get("norm"))
    # Search only the detector-selected reaction span. Nearby transcript
    # context frequently contains a police phrase spoken by somebody else and
    # caused 5/24 false-positive matches in the first source-disjoint audit.
    text = " ".join(
        str(reaction.get(field) or "")
        for field in ("phrase", "matched_text")
    )
    cue_ids: list[str] = []
    if tag in AUTHORITY_TAGS:
        cue_ids.append(f"tag:{tag}")
    if tag == "command" and norm in AUTHORITY_NORMS:
        cue_ids.append(f"conjunction:command+norm:{norm}")
    cue_ids.extend(
        f"phrase:{name}" for name, pattern in AUTHORITY_PATTERNS if pattern.search(text)
    )
    return {
        "authority_reaction_cue": bool(cue_ids),
        "cue_ids": sorted(set(cue_ids)),
        "tag": reaction.get("tag"),
        "norm": reaction.get("norm"),
        "phrase": reaction.get("phrase") or reaction.get("matched_text"),
    }


def score_metadata(path: Path) -> list[dict[str, Any]]:
    metadata = json.loads(path.read_text())
    uid = str(metadata.get("video_id") or path.parent.name)
    scene = ((metadata.get("provenance") or {}).get("scene") or {})
    by_clip: dict[int, list[dict[str, Any]]] = {}
    for reaction in metadata.get("reactions") or []:
        try:
            clip_idx = int(reaction["clip_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        by_clip.setdefault(clip_idx, []).append(score_reaction(reaction))

    rows = []
    clip_indices = sorted(
        set(range(int(metadata.get("n_clips") or 0))) | set(by_clip)
    )
    for clip_idx in clip_indices:
        reactions = by_clip.get(clip_idx, [])
        cue_ids = sorted(
            {cue for reaction in reactions for cue in reaction["cue_ids"]}
        )
        rows.append(
            {
                "item_id": f"witnessed:{uid}:clip_{clip_idx}",
                "uid": uid,
                "clip_idx": clip_idx,
                "clip_name": f"clip_{clip_idx}.mp4",
                "metadata_path": str(path),
                "reactor_role_provenance": scene.get("reactor_role"),
                "authority_reaction_cue": bool(cue_ids),
                "authority_cue_ids": cue_ids,
                "reaction_observations": reactions,
                "strict_witnessed_text_cue_disposition": (
                    "exclude_strict_preserve_scene_candidate"
                    if cue_ids
                    else "no_authority_cue"
                ),
                "corpus_disposition": None,
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen output: {args.out}")

    rows: list[dict[str, Any]] = []
    metadata_paths = sorted(args.hits.glob("*/metadata.json"))
    for path in metadata_paths:
        rows.extend(score_metadata(path))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    cue_counts = Counter(
        cue for row in rows for cue in row["authority_cue_ids"]
    )
    summary = {
        "schema_version": 3,
        "kind": "witnessed_authority_reaction_text_cues_v3",
        "rule_version": "authority_enforcement_exact_span_v3",
        "metadata_files": len(metadata_paths),
        "clip_records": len(rows),
        "authority_cue_records": sum(row["authority_reaction_cue"] for row in rows),
        "cue_counts": dict(sorted(cue_counts.items())),
        "output_sha256": sha256_file(args.out),
        "policy": "shadow_only_non_destructive",
        "manual_match_and_miss_audit_required": True,
        "corpus_mutated": False,
    }
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
