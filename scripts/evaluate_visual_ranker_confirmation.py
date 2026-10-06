#!/usr/bin/env python3
"""Evaluate frozen visual-ranker score bands after blind+dense review."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_reviews(
    blind: list[dict[str, Any]],
    dense: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = {row["item_id"]: row for row in blind}
    if len(merged) != len(blind):
        raise ValueError("duplicate blind item_id")
    for row in dense:
        if row["item_id"] not in merged:
            raise ValueError(f"dense item absent from blind set: {row['item_id']}")
        merged[row["item_id"]] = {**merged[row["item_id"]], **row}
    return merged


def truth_for(pillar: str, row: dict[str, Any]) -> dict[str, int]:
    if pillar == "instructional":
        return {
            "visual_demo": int(row["visual_demo_present"] == "yes"),
            "social_interaction_scene": int(
                row["social_interaction_scene_visible"] == "yes"
            ),
        }
    return {
        "human_event": int(row["human_event_visible"] == "yes"),
        "social_interaction_scene": int(
            row["social_interaction_scene_visible"] == "yes"
        ),
        "organic_witnessed_candidate": int(
            row["witnessed_candidate_blind"] == "yes"
        ),
    }


def target_metrics(
    rows: list[dict[str, Any]],
    target: str,
) -> dict[str, Any]:
    y = np.asarray([row["truth"][target] for row in rows], dtype=int)
    score = np.asarray([row["visual_score"] for row in rows], dtype=float)
    bands: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        bands[row["score_band"]].append(row["truth"][target])
    return {
        "items": len(rows),
        "positives": int(y.sum()),
        "roc_auc": (
            float(roc_auc_score(y, score))
            if len(set(y.tolist())) == 2
            else None
        ),
        "average_precision": (
            float(average_precision_score(y, score)) if y.sum() else None
        ),
        "bands": {
            band: {
                "items": len(values),
                "positives": int(sum(values)),
                "rate": float(np.mean(values)),
            }
            for band, values in sorted(bands.items())
        },
        "positive_item_ids": [
            row["item_id"] for row in rows if row["truth"][target]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", required=True, type=Path)
    parser.add_argument("--blind-review", required=True, type=Path)
    parser.add_argument("--dense-review", type=Path)
    parser.add_argument(
        "--pillar",
        required=True,
        choices=("instructional", "witnessed"),
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    reviews = merge_reviews(
        read_jsonl(args.blind_review),
        read_jsonl(args.dense_review),
    )
    sealed = read_jsonl(args.sealed)
    if set(reviews) != {row["item_id"] for row in sealed}:
        raise ValueError("sealed selection and reviews are not identity-exact")
    joined = []
    for row in sealed:
        if row["pillar"] != args.pillar:
            raise ValueError(f"wrong pillar: {row['item_id']}")
        joined.append(
            {
                **row,
                "truth": truth_for(args.pillar, reviews[row["item_id"]]),
            }
        )
    targets = sorted(joined[0]["truth"])
    report = {
        "protocol": (
            "new_source_disjoint_rank_bands_blind_12_frames_dense_36_if_ambiguous"
        ),
        "policy": "confirmation_audit_only_no_corpus_disposition",
        "pillar": args.pillar,
        "items": len(joined),
        "targets": {
            target: target_metrics(joined, target) for target in targets
        },
    }
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
