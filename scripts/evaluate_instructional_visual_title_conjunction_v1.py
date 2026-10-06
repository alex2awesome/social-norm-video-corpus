#!/usr/bin/env python3
"""Measure a discovered instructional visual-consensus/title conjunction.

This is development analysis only.  It joins already completed manual audits,
does not authorize any operational use, and emits a frozen rule for a future
source-disjoint transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_instructional_scene_title_cues import title_cues
else:
    from evaluate_instructional_scene_title_cues import title_cues


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0", ""}:
        return False
    raise ValueError(f"invalid boolean-like value: {value!r}")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if not n:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def metric(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    selected = [row for row in rows if row["visual_title_conjunction"]]
    positives = [row for row in rows if row[target]]
    tp = sum(row[target] for row in selected)
    fp = len(selected) - tp
    fn = len(positives) - tp
    return {
        "n": len(rows),
        "manual_positives": len(positives),
        "selected": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / len(selected) if selected else None,
        "precision_wilson_95": wilson(tp, len(selected)),
        "recall": tp / len(positives) if positives else None,
    }


def evaluate(
    semantic_rows: list[dict[str, Any]],
    demo_rows: list[dict[str, Any]],
    v23_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    semantic = {str(row["item_id"]): row for row in semantic_rows}
    demo = {str(row["item_id"]): row for row in demo_rows}
    v23 = {str(row["item_id"]): row for row in v23_rows}
    if any(len(table) != len(rows) for table, rows in (
        (semantic, semantic_rows), (demo, demo_rows), (v23, v23_rows)
    )):
        raise ValueError("duplicate item_id")
    if not (set(semantic) == set(demo) == set(v23)):
        raise ValueError("artifacts do not cover the same items")
    rows = []
    for item_id in sorted(semantic):
        metadata = semantic[item_id]
        visual = v23[item_id]
        manual = demo[item_id]
        cue = bool(title_cues(str(metadata.get("title") or ""))["scene_title_candidate"])
        consensus = boolean(visual.get("visual_consensus"))
        rows.append({
            "item_id": item_id,
            "uid": str(metadata["uid"]),
            "scene_title_candidate": cue,
            "dual_visual_episode_consensus": consensus,
            "visual_title_conjunction": cue and consensus,
            "manual_visual_demo": boolean(manual["manual_visual_demo"]),
            "manual_usable_original_label": boolean(visual["manual_usable_demo"]),
        })
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError("development cohort is not source-disjoint")
    report = {
        "kind": "instructional_visual_title_conjunction_development_v1",
        "items": len(rows),
        "manual_review_complete": True,
        "model_outputs_previously_manually_reviewed": 400,
        "frozen_rule": {
            "all": ["dual_visual_episode_consensus", "scene_title_candidate"],
        },
        "metrics": {
            "visual_demo_any_valid_medium": metric(rows, "manual_visual_demo"),
            "usable_original_label": metric(rows, "manual_usable_original_label"),
        },
        "status": "discovered_on_this_cohort_requires_fresh_source_disjoint_transfer",
        "allowed_uses": [],
        "automatic_acceptance": False,
        "automatic_rejection": False,
        "corpus_mutated": False,
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--demo-evaluated", type=Path, required=True)
    parser.add_argument("--v23-evaluated", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    if args.rows_out.exists() or args.summary_out.exists():
        raise SystemExit("refusing to overwrite discovery artifacts")
    rows, report = evaluate(
        read_jsonl(args.semantic), read_jsonl(args.demo_evaluated),
        read_jsonl(args.v23_evaluated),
    )
    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    args.rows_out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report["artifact_sha256"] = {
        "semantic": sha256(args.semantic),
        "demo_evaluated": sha256(args.demo_evaluated),
        "v23_evaluated": sha256(args.v23_evaluated),
        "normalized_rows": sha256(args.rows_out),
    }
    args.summary_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
