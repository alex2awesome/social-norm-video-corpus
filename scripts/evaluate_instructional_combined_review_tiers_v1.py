#!/usr/bin/env python3
"""Evaluate the joint ordering of two audited instructional review signals.

This does not fit a label model. It explicitly measures whether retro scene-scan
provenance and non-explanation polarity provide independent votes or a nested
ranking. Every manually audited demo remains in the queue.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


NON_EXPLANATION = {"violation", "correct", "contrast"}
TIERS = ("both", "retro_only", "polarity_only", "neither")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_cohort(
    name: str, selection: list[dict[str, Any]], ledger: list[dict[str, str]]
) -> list[dict[str, Any]]:
    manual = {int(row["audit_index"]): row for row in ledger}
    expected = {int(row["audit_index"]) for row in selection}
    if len(manual) != len(ledger) or set(manual) != expected:
        raise ValueError(f"{name}: manual ledger does not exactly cover selection")
    uids = [str(row["uid"]) for row in selection]
    if len(uids) != len(set(uids)):
        raise ValueError(f"{name}: selection is not source-disjoint")
    rows = []
    for item in selection:
        retro = str(item.get("query_source") or "") == "retro_instr_scan"
        non_explanation = str(item.get("polarity") or "") in NON_EXPLANATION
        tier = (
            "both" if retro and non_explanation else
            "retro_only" if retro else
            "polarity_only" if non_explanation else
            "neither"
        )
        rows.append({
            "cohort": name,
            "item_id": str(item["item_id"]),
            "uid": str(item["uid"]),
            "query_source": str(item.get("query_source") or ""),
            "polarity": str(item.get("polarity") or ""),
            "tier": tier,
            "visual_demo": manual[int(item["audit_index"])]["visual_demo"] == "yes",
        })
    return rows


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = sum(row["visual_demo"] for row in rows)
    return {
        "items": len(rows),
        "visual_demos": positives,
        "visual_demo_rate": positives / len(rows) if rows else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": len(rows),
        "visual_demos": sum(row["visual_demo"] for row in rows),
        "tiers": {tier: metric([row for row in rows if row["tier"] == tier]) for tier in TIERS},
    }


def evaluate(cohorts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summaries = {name: summarize(rows) for name, rows in cohorts.items()}
    all_rows = [row for rows in cohorts.values() for row in rows]
    combined = summarize(all_rows)
    checks: dict[str, bool] = {
        "minimum_combined_both_items": combined["tiers"]["both"]["items"] >= 20,
        "minimum_combined_both_precision": (
            combined["tiers"]["both"]["visual_demo_rate"] is not None
            and combined["tiers"]["both"]["visual_demo_rate"] >= 0.60
        ),
        "signals_are_nested_in_audited_cohorts": combined["tiers"]["retro_only"]["items"] == 0,
    }
    for name, summary in summaries.items():
        both = summary["tiers"]["both"]["visual_demo_rate"]
        polarity = summary["tiers"]["polarity_only"]["visual_demo_rate"]
        neither = summary["tiers"]["neither"]["visual_demo_rate"]
        checks[f"{name}_minimum_both_items"] = summary["tiers"]["both"]["items"] >= 7
        checks[f"{name}_monotonic_both_over_polarity"] = (
            both is not None and polarity is not None and both > polarity
        )
        checks[f"{name}_monotonic_polarity_over_neither"] = (
            polarity is not None and neither is not None and polarity > neither
        )
    passed = all(checks.values())
    positives = combined["visual_demos"]
    both_tp = combined["tiers"]["both"]["visual_demos"]
    return {
        "kind": "instructional_combined_review_tiers_v1",
        "cohorts": summaries,
        "combined": combined,
        "combined_both_recall": both_tp / positives if positives else None,
        "signal_dependence": (
            "retro_is_nested_within_non_explanation_in_both_audited_cohorts; "
            "do_not_treat_as_independent_snorkel_votes"
        ),
        "gate_checks": checks,
        "review_tiering_promoted": passed,
        "allowed_use": "manual_review_order_only" if passed else "none",
        "tier_order": ["both", "polarity_only", "neither"] if passed else [],
        "retro_only_policy": "retain_existing_retro_priority_but_report_out_of_support",
        "automatic_acceptance": False,
        "automatic_rejection": False,
        "corpus_mutated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", nargs=3, action="append", metavar=("NAME", "SELECTION", "LEDGER"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    cohorts = {}
    hashes = {}
    for name, selection_value, ledger_value in args.cohort:
        selection = Path(selection_value)
        ledger = Path(ledger_value)
        cohorts[name] = load_cohort(name, read_jsonl(selection), read_tsv(ledger))
        hashes[name] = {"selection": sha256(selection), "ledger": sha256(ledger)}
    report = evaluate(cohorts)
    report["artifact_sha256"] = hashes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["review_tiering_promoted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
