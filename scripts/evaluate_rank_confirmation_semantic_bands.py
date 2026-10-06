#!/usr/bin/env python3
"""Evaluate frozen post-reveal outcomes by visual-ranker score band.

This is an audit report only. It joins sealed scores to manual judgments and,
for witnessed clips, can evaluate the already-audited title staging cue as a
second-stage exclusion. It never changes corpus metadata or routing.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def keyed(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result = {str(row[field]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {field}")
    return result


def binary_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    positives = sum(row.get(field) == "yes" for row in rows)
    return {
        "items": len(rows),
        "positives": positives,
        "rate": positives / len(rows) if rows else None,
    }


def selection_summary(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    outcome: str,
) -> dict[str, Any]:
    positives = sum(row.get(outcome) == "yes" for row in selected)
    total_positives = sum(row.get(outcome) == "yes" for row in rows)
    return {
        "selected": len(selected),
        "true_positive": positives,
        "precision": positives / len(selected) if selected else None,
        "recall": positives / total_positives if total_positives else None,
    }


def evaluate(
    sealed: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    pillar: str,
    blind: list[dict[str, Any]] | None = None,
    dense: list[dict[str, Any]] | None = None,
    staging: list[dict[str, Any]] | None = None,
    production_adjudications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    review_by_id = keyed(reviews, "item_id")
    if set(review_by_id) != {str(row["item_id"]) for row in sealed}:
        raise ValueError("sealed and post-reveal review coverage differ")
    visual_by_id = keyed(blind or [], "item_id")
    for row in dense or []:
        if str(row["item_id"]) not in visual_by_id:
            raise ValueError("dense item absent from blind review")
        visual_by_id[str(row["item_id"])] = row
    staging_by_uid = keyed(staging or [], "uid")
    production_by_id = keyed(production_adjudications or [], "item_id")
    strict_field = (
        "strict_instructional_pass"
        if pillar == "instructional"
        else "strict_witnessed_pass"
    )
    joined = []
    for score in sealed:
        item_id = str(score["item_id"])
        row = {**score, **review_by_id[item_id]}
        row.update(visual_by_id.get(item_id, {}))
        production = production_by_id.get(item_id)
        if production:
            value = production.get("witnessed_candidate_post_reveal")
            if value not in {"yes", "no", "uncertain"}:
                raise ValueError(
                    f"{item_id}: invalid production adjudication"
                )
            row["witnessed_candidate_post_reveal"] = value
        elif "witnessed_candidate_blind" in row:
            row["witnessed_candidate_post_reveal"] = row[
                "witnessed_candidate_blind"
            ]
        cue = staging_by_uid.get(str(score["uid"]), {}).get("title_staging_cue")
        row["title_staging_cue"] = cue if isinstance(cue, bool) else None
        joined.append(row)

    bands = {}
    for band in sorted({str(row["score_band"]) for row in joined}):
        subset = [row for row in joined if row["score_band"] == band]
        bands[band] = {
            **binary_summary(subset, strict_field),
            "routes": dict(
                sorted(Counter(str(row["recoverable_route"]) for row in subset).items())
            ),
        }
        if pillar == "instructional":
            bands[band]["visual_demo"] = binary_summary(
                subset, "visual_demo_present"
            )
            bands[band]["social_norm"] = binary_summary(
                subset, "assigned_norm_is_social_norm"
            )
        elif visual_by_id:
            bands[band]["organic_candidate"] = binary_summary(
                subset, "witnessed_candidate_post_reveal"
            )

    selections = {
        "top_quintile": selection_summary(
            joined,
            [row for row in joined if row["score_band"] == "top_quintile"],
            strict_field,
        )
    }
    if pillar == "witnessed" and staging_by_uid:
        top_non_title = [
            row
            for row in joined
            if row["score_band"] == "top_quintile"
            and row["title_staging_cue"] is False
        ]
        selections["top_quintile_without_title_staging_cue"] = (
            selection_summary(joined, top_non_title, strict_field)
        )
        if visual_by_id:
            selections["top_quintile_without_title_staging_cue"][
                "organic_candidate"
            ] = selection_summary(
                joined, top_non_title, "witnessed_candidate_post_reveal"
            )
        selections["title_staging_cue_coverage"] = {
            "known": sum(row["title_staging_cue"] is not None for row in joined),
            "positive": sum(row["title_staging_cue"] is True for row in joined),
        }
        selections["post_reveal_production_adjudications"] = len(
            production_by_id
        )

    return {
        "schema_version": 1,
        "kind": "rank_confirmation_semantic_bands",
        "pillar": pillar,
        "policy": "manual_audit_only_no_corpus_mutation",
        "items": len(joined),
        "strict": binary_summary(joined, strict_field),
        "bands": bands,
        "selections": selections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--post-review", type=Path, required=True)
    parser.add_argument(
        "--pillar", choices=("instructional", "witnessed"), required=True
    )
    parser.add_argument("--blind-review", type=Path)
    parser.add_argument("--dense-review", type=Path)
    parser.add_argument("--staging-cues", type=Path)
    parser.add_argument("--production-adjudications", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    report = evaluate(
        read_jsonl(args.sealed),
        read_jsonl(args.post_review),
        args.pillar,
        read_jsonl(args.blind_review),
        read_jsonl(args.dense_review),
        read_jsonl(args.staging_cues),
        read_jsonl(args.production_adjudications),
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
