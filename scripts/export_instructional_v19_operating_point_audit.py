#!/usr/bin/env python3
"""Export manually labeled V18 outputs for a train-fit V19 operating point."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


MANUAL_TARGET_KEYS = {
    "visual": "manual_visual_demo",
    "usable": "manual_relabel_usable",
    "exact": "manual_exact_social_norm",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def export_rows(
    report: dict[str, Any],
    manifest: list[dict[str, Any]],
    manual: list[dict[str, Any]],
    group: str,
    target: str,
    minimum_predictions: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = report["model_results"][group][target]
    point = result["best_precision_operating_points"][
        str(minimum_predictions)
    ]
    threshold = float(point["threshold_fit_train_oof"])
    test = [
        row for row in manifest
        if row["audit_cohort"] == "v18"
    ]
    scores = list(map(float, result["v18_scores"]))
    if len(test) != len(scores):
        raise ValueError("V18 score count does not match manifest")
    manual_by_item = {str(row["item_id"]): row for row in manual}
    selected = []
    for source, score in zip(test, scores):
        if score < threshold:
            continue
        item_id = str(source["item_id"])
        if item_id not in manual_by_item:
            raise ValueError(f"missing manual row: {item_id}")
        selected.append(
            {
                **source,
                **manual_by_item[item_id],
                "shadow_feature_group": group,
                "shadow_target": target,
                "shadow_score": score,
                "threshold_fit_v15_v17_oof": threshold,
            }
        )
    manual_key = MANUAL_TARGET_KEYS[target]
    positives = sum(bool(row[manual_key]) for row in selected)
    summary = {
        "kind": "instructional_v19_operating_point_manual_output_audit",
        "status": "posthoc_development_not_promotion",
        "policy": "read_only_no_keep_reject_or_corpus_mutation",
        "feature_group": group,
        "target": target,
        "minimum_train_predictions": minimum_predictions,
        "threshold_fit_v15_v17_oof": threshold,
        "selected_v18": len(selected),
        "manual_positives": positives,
        "manual_precision": positives / len(selected) if selected else None,
        "manual_failure_modes": dict(
            Counter(
                str(row.get("manual_failure_mode") or "none")
                for row in selected
                if not bool(row[manual_key])
            )
        ),
        "reported_train_metrics": point["train_oof_metrics"],
        "reported_v18_metrics": point["v18_transfer_metrics"],
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--target", choices=("visual", "usable", "exact"), required=True)
    parser.add_argument("--minimum-predictions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    selected, summary = export_rows(
        json.loads(args.benchmark.read_text()),
        read_jsonl(args.manifest),
        read_jsonl(args.manual),
        args.group,
        args.target,
        args.minimum_predictions,
    )
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    )
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
