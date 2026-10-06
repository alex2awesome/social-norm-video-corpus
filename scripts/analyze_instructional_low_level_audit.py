#!/usr/bin/env python3
"""Diagnose cheap visual features on a frozen manual instructional audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


FEATURES = (
    "face_count_mean",
    "face_present_fraction",
    "hard_cut_fraction",
    "histogram_delta_mean",
    "motion_max",
    "motion_mean",
    "multiple_faces_fraction",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def analyze(
    manifest: list[dict[str, Any]],
    ledger: list[dict[str, str]],
    feature_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_index = {int(row["audit_index"]): row for row in ledger}
    by_item = {
        str(row["item_id"]): row
        for row in feature_rows
        if row.get("error") in (None, "") and isinstance(row.get("low_level"), dict)
    }
    if len(by_index) != len(manifest):
        raise ValueError("ledger must exactly cover manifest")
    joined = []
    for source in manifest:
        index = int(source["audit_index"])
        manual = by_index[index]
        if manual["candidate_id"] != source["candidate_id"]:
            raise ValueError(f"candidate mismatch at {index}")
        item_id = str(source["item_id"])
        if item_id not in by_item:
            raise ValueError(f"missing features for {item_id}")
        joined.append(
            (
                manual["visual_demo"].strip().lower() in {"y", "yes"},
                by_item[item_id]["low_level"],
            )
        )
    labels = np.asarray([label for label, _ in joined], dtype=bool)
    results = {}
    for name in FEATURES:
        values = np.asarray([float(features[name]) for _, features in joined])
        raw_ap = float(average_precision_score(labels, values))
        inverse_ap = float(average_precision_score(labels, -values))
        raw_auc = float(roc_auc_score(labels, values))
        direction = "higher" if raw_ap >= inverse_ap else "lower"
        results[name] = {
            "positive_mean": float(values[labels].mean()),
            "negative_mean": float(values[~labels].mean()),
            "raw_average_precision": raw_ap,
            "inverse_average_precision": inverse_ap,
            "best_univariate_average_precision": max(raw_ap, inverse_ap),
            "raw_roc_auc": raw_auc,
            "best_direction": direction,
        }
    return {
        "kind": "instructional_low_level_fresh_transfer_diagnostic",
        "status": "diagnostic_only_no_rule_promotion",
        "policy": "read_only_no_keep_reject_or_corpus_mutation",
        "items": len(joined),
        "visual_demos": int(labels.sum()),
        "non_demos": int((~labels).sum()),
        "positive_prevalence": float(labels.mean()),
        "features": results,
        "best_feature": max(
            results,
            key=lambda name: results[name]["best_univariate_average_precision"],
        ),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        read_jsonl(args.manifest), read_tsv(args.ledger), read_jsonl(args.features)
    )
    report["artifact_sha256"] = {
        "manifest": sha256(args.manifest),
        "ledger": sha256(args.ledger),
        "features": sha256(args.features),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
