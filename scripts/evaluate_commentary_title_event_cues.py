#!/usr/bin/env python3
"""Evaluate a frozen blind/post-reveal title-event cue audit."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STRICT = {"title_labeled_visual_event", "instructional_reroute"}
FOLLOWUP = {"crop_review", "dense_review", "audio_or_dense_review"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique(rows: list[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key}")
    return result


def wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def group_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict = sum(row["disposition"] in STRICT for row in rows)
    followup = sum(row["disposition"] in FOLLOWUP for row in rows)
    return {
        "items": len(rows),
        "strict_usable": strict,
        "followup": followup,
        "strict_rate": strict / len(rows) if rows else None,
        "strict_or_followup_rate": (strict + followup) / len(rows) if rows else None,
    }


def evaluate(
    blind_manifest: list[dict[str, Any]],
    blind_review: list[dict[str, Any]],
    sealed: list[dict[str, Any]],
    post: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    corpus_candidates: int,
) -> dict[str, Any]:
    blind_by_index = unique(blind_manifest, "audit_index")
    review_by_index = unique(blind_review, "audit_index")
    sealed_by_index = unique(sealed, "audit_index")
    post_by_index = unique(post, "audit_index")
    indices = set(blind_by_index)
    if not (
        indices == set(review_by_index) == set(sealed_by_index) == set(post_by_index)
    ):
        raise ValueError("blind, sealed, and post-reveal coverage must match")
    candidate_by_item = unique(candidates, "item_id")

    rows = []
    for index in sorted(indices):
        blind = blind_by_index[index]
        reveal = sealed_by_index[index]
        if blind["item_id"] != reveal["item_id"] or blind["uid"] != reveal["uid"]:
            raise ValueError(f"sealed identity mismatch at audit_index {index}")
        candidate = candidate_by_item.get(blind["item_id"])
        if candidate is None:
            raise ValueError(f"selected item absent from candidate manifest: {blind['item_id']}")
        rows.append(
            {
                **blind,
                **review_by_index[index],
                **reveal,
                **post_by_index[index],
                "title_event_cues": candidate["title_event_cues"],
            }
        )

    strict = sum(row["disposition"] in STRICT for row in rows)
    interval = wilson(strict, len(rows))
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_band[row["score_band"]].append(row)
        for cue in row["title_event_cues"]:
            by_cue[cue].append(row)

    return {
        "kind": "commentary_title_event_cue_manual_evaluation",
        "coverage_complete": True,
        "policy": "shadow_only_no_keep_reject",
        "overall": group_report(rows),
        "blind_visual_labels": dict(
            Counter(row["social_event_visible"] for row in rows)
        ),
        "dispositions": dict(Counter(row["disposition"] for row in rows)),
        "strict_organic_or_news": sum(
            row["disposition"] == "title_labeled_visual_event" for row in rows
        ),
        "strict_instructional_reroutes": sum(
            row["disposition"] == "instructional_reroute" for row in rows
        ),
        "strict_rate_wilson_95": interval,
        "corpus_candidates": corpus_candidates,
        "estimated_strict_survivors": corpus_candidates * strict / len(rows),
        "estimated_strict_survivors_wilson_95": (
            [corpus_candidates * interval[0], corpus_candidates * interval[1]]
            if interval
            else None
        ),
        "by_activity_band": {
            key: group_report(value) for key, value in sorted(by_band.items())
        },
        "by_title_cue": {
            key: group_report(value) for key, value in sorted(by_cue.items())
        },
        "strict_audit_indices": [
            row["audit_index"] for row in rows if row["disposition"] in STRICT
        ],
        "followup_audit_indices": [
            row["audit_index"] for row in rows if row["disposition"] in FOLLOWUP
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--blind-review", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--corpus-candidates", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        load_jsonl(args.blind_manifest),
        load_jsonl(args.blind_review),
        load_jsonl(args.sealed),
        load_jsonl(args.post),
        load_jsonl(args.candidates),
        args.corpus_candidates,
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["overall"], sort_keys=True))


if __name__ == "__main__":
    main()
