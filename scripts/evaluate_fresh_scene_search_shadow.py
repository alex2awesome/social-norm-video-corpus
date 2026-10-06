#!/usr/bin/env python3
"""Evaluate a complete visual review of a frozen fresh-search sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rendered = [row for row in rows if row["artifact_status"] == "rendered"]
    strict_yes = sum(row["strict_pillar_candidate"] == "yes" for row in rendered)
    strict_uncertain = sum(
        row["strict_pillar_candidate"] == "uncertain" for row in rendered
    )
    target_yes = sum(row["target_query_match"] == "yes" for row in rendered)
    visual_yes = sum(row["visual_scene_candidate"] == "yes" for row in rendered)
    return {
        "selected": len(rows),
        "rendered": len(rendered),
        "download_failed": len(rows) - len(rendered),
        "visual_scene_yes": visual_yes,
        "target_query_yes": target_yes,
        "strict_yes": strict_yes,
        "strict_uncertain": strict_uncertain,
        "strict_lower_bound_rate": strict_yes / len(rendered) if rendered else None,
        "strict_upper_bound_rate": (
            (strict_yes + strict_uncertain) / len(rendered) if rendered else None
        ),
    }


def evaluate(manifest: dict[str, Any], reviews: list[dict[str, Any]]) -> dict[str, Any]:
    records = manifest["records"]
    by_ordinal = {row["ordinal"]: row for row in records}
    reviews_by_ordinal = {row["ordinal"]: row for row in reviews}
    if len(by_ordinal) != len(records) or len(reviews_by_ordinal) != len(reviews):
        raise ValueError("duplicate ordinal in manifest or manual review")
    if set(by_ordinal) != set(reviews_by_ordinal):
        missing = sorted(set(by_ordinal) - set(reviews_by_ordinal))
        extra = sorted(set(reviews_by_ordinal) - set(by_ordinal))
        raise ValueError(f"incomplete review: missing={missing}, extra={extra}")

    joined: list[dict[str, Any]] = []
    for ordinal in sorted(by_ordinal):
        source = by_ordinal[ordinal]
        review = reviews_by_ordinal[ordinal]
        for field in ("uid", "artifact_status", "group"):
            if source[field] != review[field]:
                raise ValueError(
                    f"ordinal {ordinal}: {field} mismatch "
                    f"{source[field]!r} != {review[field]!r}"
                )
        joined.append({**source, **review})

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_group[row["group"]].append(row)
        by_query[row["query"]].append(row)

    rendered = [row for row in joined if row["artifact_status"] == "rendered"]
    unique_title_rows: list[dict[str, Any]] = []
    seen_titles: set[tuple[str, str]] = set()
    for row in rendered:
        key = (row["query"], " ".join(row["title"].lower().split()))
        if key not in seen_titles:
            unique_title_rows.append(row)
            seen_titles.add(key)

    return {
        "kind": "fresh_scene_search_shadow_manual_evaluation",
        "coverage_complete": True,
        "overall": summarize(joined),
        "unique_query_title_overall": summarize(unique_title_rows),
        "by_group": {
            key: summarize(rows) for key, rows in sorted(by_group.items())
        },
        "by_query": {
            key: summarize(rows) for key, rows in sorted(by_query.items())
        },
        "strict_yes_ordinals": [
            row["ordinal"]
            for row in rendered
            if row["strict_pillar_candidate"] == "yes"
        ],
        "strict_uncertain_ordinals": [
            row["ordinal"]
            for row in rendered
            if row["strict_pillar_candidate"] == "uncertain"
        ],
        "reroute_candidates": [
            {
                "ordinal": row["ordinal"],
                "uid": row["uid"],
                "from": row["group"],
                "to": row["reroute_candidate"],
            }
            for row in rendered
            if row.get("reroute_candidate")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("manual_review", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = evaluate(
        json.loads(args.manifest.read_text()),
        load_jsonl(args.manual_review),
    )
    report["manifest_sha256"] = sha256(args.manifest)
    report["manual_review_sha256"] = sha256(args.manual_review)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
