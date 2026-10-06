#!/usr/bin/env python3
"""Benchmark scalable visual features on the frozen fresh-query audit.

The UID hash split is source-disjoint.  This is a diagnostic benchmark, not a
rule promotion step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_visual_scene_scores import (
        best_threshold,
        evaluate_multivariate,
        flatten_baseline,
        split,
    )
else:
    from evaluate_visual_scene_scores import (
        best_threshold,
        evaluate_multivariate,
        flatten_baseline,
        split,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _indexed(rows: list[dict[str, Any]], name: str) -> dict[int, dict[str, Any]]:
    result = {}
    for row in rows:
        index = row.get("audit_index")
        if not isinstance(index, int) or index in result:
            raise ValueError(f"{name}: invalid or duplicate audit_index {index!r}")
        result[index] = row
    return result


def _score_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        item_id = row.get("item_id")
        if not item_id:
            raise ValueError("baseline row missing item_id")
        if row.get("error") is None:
            result[str(item_id)] = row
    return result


def evaluate(
    sealed_rows: list[dict[str, Any]],
    blind_rows: list[dict[str, Any]],
    post_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sealed = _indexed(sealed_rows, "sealed")
    blind = _indexed(blind_rows, "blind")
    post = _indexed(post_rows, "post")
    if set(sealed) != set(blind) or set(sealed) != set(post):
        raise ValueError("manual audit coverage mismatch")
    scores = _score_index(baseline_rows)
    missing = [
        row["item_id"] for row in sealed.values() if row["item_id"] not in scores
    ]
    if missing:
        raise ValueError(f"missing successful baseline rows: {missing}")

    joined = []
    for index in sorted(sealed):
        source = sealed[index]
        joined.append(
            {
                "item_id": source["item_id"],
                "uid": source["uid"],
                "pillar": source["source_modality"],
                "blind": blind[index]["any_pillar_visual_candidate"],
                "strict": bool(post[index]["strict_pillar_pass"]),
                "features": flatten_baseline(scores[source["item_id"]]),
            }
        )

    def target_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
        feature_names = sorted({name for row in rows for name in row["features"]})
        feature_reports = [
            report
            for name in feature_names
            if (report := best_threshold(rows, name)) is not None
        ]
        feature_reports.sort(
            key=lambda report: report["test"]["balanced_accuracy"], reverse=True
        )
        return {
            "items": len(rows),
            "sources": len({row["uid"] for row in rows}),
            "positives": sum(row["gold"] for row in rows),
            "source_disjoint_split": {
                name: {
                    "items": sum(split(row["uid"]) == name for row in rows),
                    "sources": len(
                        {row["uid"] for row in rows if split(row["uid"]) == name}
                    ),
                    "positives": sum(
                        row["gold"] and split(row["uid"]) == name for row in rows
                    ),
                }
                for name in ("train", "test")
            },
            "atomic_features": feature_reports,
            "multivariate": evaluate_multivariate(rows),
        }

    visual_rows = [
        {**row, "gold": row["blind"] == "yes"}
        for row in joined
        if row["blind"] in {"yes", "no"}
    ]
    strict_rows = [{**row, "gold": row["strict"]} for row in joined]
    return {
        "kind": "fresh_scene_scalable_visual_baseline_benchmark",
        "visual_candidate_definitive": target_report(visual_rows),
        "strict_pillar_pass": target_report(strict_rows),
        "decision": {
            "automatic_promotion": False,
            "reason": (
                "Single frozen query cohort; thresholds require source-disjoint "
                "manual confirmation before any production use."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    report = evaluate(
        read_jsonl(args.sealed),
        read_jsonl(args.blind),
        read_jsonl(args.post),
        read_jsonl(args.baselines),
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
