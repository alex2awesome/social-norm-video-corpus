#!/usr/bin/env python3
"""Evaluate v7 atomic social-gate bands against consolidated manual gold."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CANDIDATE_BANDS = {"v7_exact_candidate", "v7_relabel_candidate"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gold_status(row: dict) -> str:
    if row["decision"] == "reject":
        return "reject"
    domain = row.get("social_norm_domain")
    if row["decision"].startswith("reroute_"):
        return "noncore_reroute"
    if domain and domain != "yes" and (
        domain.startswith("broader_") or domain.startswith("health_")
    ):
        return "noncore_reroute"
    return "core_useful"


def evaluate(scores: list[dict], gold_rows: list[dict]) -> dict:
    gold = {row["item_id"]: row for row in gold_rows}
    joined = [
        (score, gold[score["item_id"]])
        for score in scores
        if score["item_id"] in gold
    ]
    candidates = [
        (score, manual)
        for score, manual in joined
        if score["derived_band"] in CANDIDATE_BANDS
    ]
    candidate_counts = Counter(gold_status(manual) for _, manual in candidates)
    core_positives = sum(
        gold_status(manual) == "core_useful" for _, manual in joined
    )
    true_positive = candidate_counts["core_useful"]
    return {
        "status": "failed_manual_calibration_do_not_promote",
        "gold_items": len(gold),
        "joined_items": len(joined),
        "candidate_bands": sorted(CANDIDATE_BANDS),
        "candidate_metrics": {
            "accepted": len(candidates),
            "core_true_positive": true_positive,
            "reject_false_positive": candidate_counts["reject"],
            "noncore_reroute": candidate_counts["noncore_reroute"],
            "core_precision": true_positive / len(candidates)
            if candidates
            else None,
            "core_recall": true_positive / core_positives
            if core_positives
            else None,
        },
        "band_counts_all_scores": dict(
            sorted(Counter(row["derived_band"] for row in scores).items())
        ),
        "candidate_records": [
            {
                "item_id": score["item_id"],
                "band": score["derived_band"],
                "gold_status": gold_status(manual),
                "gold_decision": manual["decision"],
                "gold_evidence": manual.get("evidence")
                or manual.get("adjudication_reason"),
            }
            for score, manual in candidates
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(load_jsonl(args.scores), load_jsonl(args.gold))
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["candidate_metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
