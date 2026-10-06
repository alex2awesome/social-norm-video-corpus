#!/usr/bin/env python3
"""Select manually screened commentary sources for transcript-centered review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_items(manifest: dict, reviews: list[dict]) -> list[dict]:
    visual = manifest.get("visual_samples") or []
    by_index = {int(row["audit_index"]): row for row in visual}
    if len(by_index) != len(visual):
        raise ValueError("duplicate audit_index in source manifest")
    review_by_index = {}
    for row in reviews:
        index = int(row["audit_index"])
        if index in review_by_index:
            raise ValueError(f"duplicate review audit_index: {index}")
        review_by_index[index] = row
    if set(by_index) != set(review_by_index):
        raise ValueError("manual review must cover the complete source manifest")

    selected = []
    for index in sorted(by_index):
        source = by_index[index]
        review = review_by_index[index]
        if source["uid"] != review["uid"]:
            raise ValueError(f"uid mismatch at audit_index {index}")
        if not review["dense_followup"]:
            continue
        statement = str(source.get("statement") or "").strip()
        if not statement:
            raise ValueError(f"dense follow-up lacks statement quote: {source['uid']}")
        selected.append(
            {
                "source_audit_index": index,
                "uid": source["uid"],
                "title": source.get("title"),
                "normalized_behavior": source.get("query") or source.get("norm"),
                "normalized_norm": source.get("norm"),
                "behavior_evidence_quote": statement,
                "stance_evidence_quote": statement,
                "source_screen": review["source_screen"],
                "source_evidence": review["evidence"],
            }
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    items = build_items(
        json.loads(args.manifest.read_text()),
        load_jsonl(args.manual_review),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"items": items}, indent=2) + "\n")
    print(json.dumps({"items": len(items), "out": str(args.out)}))


if __name__ == "__main__":
    main()
