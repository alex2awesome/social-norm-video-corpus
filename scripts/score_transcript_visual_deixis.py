#!/usr/bin/env python3
"""Localize narrated visual-event windows from timestamped transcripts.

The score is a temporal prior only. Phrases such as "watch as" and "you can
see" can identify where a news package displays its source footage, but they
never prove that the claimed action is visually established.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEIXIS = re.compile(
    r"\b(?:"
    r"you can see|as you can see|watch as|watch this|take a look|"
    r"shows? the moment|video shows?|footage shows?|"
    r"surveillance (?:video|footage) shows?|"
    r"this is the (?:recording|video|footage)|"
    r"caught on (?:camera|video|cctv)|"
    r"on camera,? you can see"
    r")\b",
    re.IGNORECASE,
)
ACTION = re.compile(
    r"\b(?:"
    r"attack(?:s|ed|ing)?|assault(?:s|ed|ing)?|beat(?:s|en|ing)?|"
    r"hit(?:s|ting)?|strik(?:e|es|ing|uck)|kick(?:s|ed|ing)?|"
    r"punch(?:es|ed|ing)?|harass(?:es|ed|ing|ment)?|"
    r"steal(?:s|ing|stole|stolen)?|tak(?:e|es|ing)|remov(?:e|es|ing)|"
    r"rob(?:s|bed|bing|bery)?|confront(?:s|ed|ing|ation)?|"
    r"fight(?:s|ing)?|brawl(?:s|ing)?|bully(?:ing|ied)?|"
    r"abus(?:e|es|ed|ing)|slap(?:s|ped|ping)?|"
    r"shov(?:e|es|ed|ing)|spit(?:s|ting)?|"
    r"ram(?:s|med|ming)?|crash(?:es|ed|ing)?|collid(?:e|es|ed|ing)|"
    r"forc(?:e|es|ed|ing)|grab(?:s|bed|bing)?|hold(?:s|ing)?|"
    r"reach(?:es|ed|ing)?|stuff(?:s|ed|ing)?|cram(?:s|med|ming)?|"
    r"pick(?:s|ed|ing)? up|help(?:s|ed|ing)? (?:her|him|them)self|"
    r"pull(?:s|ed|ing)? (?:her |his |their )?hair|snatch(?:es|ed|ing)?|"
    r"slam(?:s|med|ming)? (?:on )?(?:the )?brakes?|"
    r"run(?:s|ning)? over|driv(?:e|es|ing|en) through|"
    r"threaten(?:s|ed|ing)?|yell(?:s|ed|ing)?|"
    r"berat(?:e|es|ed|ing)|tirade|racial slur"
    r")\b",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def localize(row: dict[str, Any], pad_sec: float = 2.0) -> dict[str, Any]:
    segments = row.get("segments") or []
    candidates = []
    for index, segment in enumerate(segments):
        text = str(segment.get("text") or "")
        if not DEIXIS.search(text):
            continue
        neighborhood = segments[max(0, index - 1) : min(len(segments), index + 2)]
        action_text = " ".join(str(item.get("text") or "") for item in neighborhood)
        # "take a look" is visual deixis, not evidence of property taking.
        action_text = DEIXIS.sub("", action_text)
        action_matches = sorted(
            {match.group(0).lower() for match in ACTION.finditer(action_text)}
        )
        if not action_matches:
            continue
        start_values = [
            item.get("start") for item in neighborhood
            if isinstance(item.get("start"), (int, float))
        ]
        end_values = [
            item.get("end") for item in neighborhood
            if isinstance(item.get("end"), (int, float))
        ]
        if not start_values or not end_values:
            continue
        candidates.append(
            {
                "start_sec": max(0.0, min(start_values) - pad_sec),
                "end_sec": max(end_values) + pad_sec,
                "deictic_segment_index": index,
                "deictic_text": text.strip(),
                "action_terms": action_matches,
            }
        )
    return {
        "audit_index": row["audit_index"],
        "candidate_id": row["candidate_id"],
        "uid": row["uid"],
        "title": row["title"],
        "visual_deixis_windows": candidates,
        "visual_deixis_window_count": len(candidates),
        "temporal_prior_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pad-sec", type=float, default=2.0)
    args = parser.parse_args()
    rows = [localize(row, args.pad_sec) for row in load_jsonl(args.transcripts)]
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary = {
        "kind": "transcript_visual_deixis_temporal_priors_v1",
        "items": len(rows),
        "items_with_windows": sum(
            bool(row["visual_deixis_windows"]) for row in rows
        ),
        "windows": sum(row["visual_deixis_window_count"] for row in rows),
        "policy": "localization_only_never_acceptance",
        "corpus_mutated": False,
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
