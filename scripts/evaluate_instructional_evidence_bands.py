#!/usr/bin/env python3
"""Evaluate blind-evidence review bands against append-only manual judgments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CANDIDATE_BANDS = {
    "blind_dual_exact_candidate",
    "blind_dual_relabel_candidate",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_gold(paths: list[Path]) -> dict[str, dict]:
    gold: dict[str, dict] = {}
    for path in paths:
        for row in load_jsonl(path):
            item_id = row["item_id"]
            if item_id in gold:
                raise ValueError(f"duplicate manual judgment: {item_id}")
            gold[item_id] = row
    return gold


def evaluate(bands: list[dict], gold: dict[str, dict]) -> dict:
    joined = []
    counts: dict[str, Counter] = {}
    for row in bands:
        item_id = row["item_id"]
        if item_id not in gold:
            continue
        usable = gold[item_id]["decision"] != "reject"
        joined.append(
            {"item_id": item_id, "band": row["band"], "gold_usable": usable}
        )
        counter = counts.setdefault(row["band"], Counter())
        counter["n"] += 1
        counter["gold_usable"] += int(usable)
        counter["gold_reject"] += int(not usable)
    predicted = [row for row in joined if row["band"] in CANDIDATE_BANDS]
    tp = sum(row["gold_usable"] for row in predicted)
    fp = len(predicted) - tp
    positives = sum(row["gold_usable"] for row in joined)
    return {
        "status": "calibration_only_requires_source_disjoint_replication",
        "manual_rows": len(gold),
        "joined_rows": len(joined),
        "candidate_bands": sorted(CANDIDATE_BANDS),
        "candidate_metrics": {
            "accepted": len(predicted),
            "tp": tp,
            "fp": fp,
            "precision": tp / len(predicted) if predicted else None,
            "recall": tp / positives if positives else None,
        },
        "bands": {band: dict(value) for band, value in sorted(counts.items())},
        "candidate_errors": [
            row for row in predicted if not row["gold_usable"]
        ],
        "records": sorted(joined, key=lambda row: row["item_id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bands", type=Path, required=True)
    parser.add_argument(
        "--manual-review", type=Path, action="append", required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = evaluate(load_jsonl(args.bands), load_gold(args.manual_review))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
