#!/usr/bin/env python3
"""Fail closed unless a blind visual audit covers every frozen item exactly once."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED = {
    "situated_social_scene": {"yes", "no", "uncertain"},
    "concrete_behavior_visible": {"yes", "no", "uncertain"},
    "affected_person_or_shared_context_visible": {"yes", "no", "uncertain"},
    "presentation_or_context_only": {"yes", "no", "uncertain"},
    "technical_or_formal_only": {"yes", "no", "uncertain"},
    "depiction_type": {
        "organic_scene",
        "enacted_scene",
        "film_tv",
        "animation",
        "screen_scenario",
        "talking_head",
        "interview",
        "news_broll",
        "graphics",
        "mixed",
        "uncertain",
    },
    "authenticity": {
        "organic",
        "hidden_camera_or_hybrid",
        "staged",
        "animation",
        "news_or_commentary",
        "uncertain",
        "na",
    },
    "dense_review_required": {"yes", "no"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def validate(
    manifest_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = [row.get("item_id") for row in manifest_rows]
    reviewed = [row.get("item_id") for row in review_rows]
    if any(not item_id for item_id in expected + reviewed):
        raise ValueError("every manifest and review row requires item_id")
    if len(set(expected)) != len(expected) or len(set(reviewed)) != len(reviewed):
        raise ValueError("duplicate item_id")
    if set(expected) != set(reviewed):
        missing = set(expected) - set(reviewed)
        extra = set(reviewed) - set(expected)
        raise ValueError(
            f"review coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )
    counts = Counter()
    for row in review_rows:
        item_id = row["item_id"]
        for field, allowed in ALLOWED.items():
            if row.get(field) not in allowed:
                raise ValueError(f"{item_id}: invalid or missing {field}")
        if not str(row.get("literal_description") or "").strip():
            raise ValueError(f"{item_id}: missing literal_description")
        ambiguous_or_positive = (
            row["situated_social_scene"] != "no"
            or row["concrete_behavior_visible"] != "no"
        )
        if ambiguous_or_positive and row["dense_review_required"] != "yes":
            raise ValueError(f"{item_id}: visible/uncertain event requires dense review")
        counts[row["situated_social_scene"]] += 1
    return {
        "manifest_items": len(expected),
        "reviewed_items": len(reviewed),
        "coverage": 1.0,
        "situated_social_scene": dict(sorted(counts.items())),
        "complete": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(read_jsonl(args.manifest), read_jsonl(args.reviews)),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
