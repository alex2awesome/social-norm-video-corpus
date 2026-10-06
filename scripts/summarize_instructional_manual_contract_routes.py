#!/usr/bin/env python3
"""Summarize deterministic instructional routes on complete manual ledgers."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = ("visual_demo", "semantic_alignment", "complete_demo", "usable_demo")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def yes(row: dict[str, str], field: str) -> bool:
    value = row.get(field)
    if value not in {"yes", "no"}:
        raise ValueError(f"{row.get('item_id')}: invalid {field}: {value!r}")
    return value == "yes"


def manual_route(row: dict[str, str]) -> str:
    visual = yes(row, "visual_demo")
    aligned = yes(row, "semantic_alignment")
    complete = yes(row, "complete_demo")
    usable = yes(row, "usable_demo")
    if usable and not (visual and aligned and complete):
        raise ValueError(f"{row.get('item_id')}: usable demo contradicts atoms")
    if usable:
        return "instructional_demo_candidate"
    if visual and complete and not aligned:
        return "instructional_demo_candidate_after_relabel"
    if visual and not complete:
        return "instructional_source_needs_recut"
    if not visual:
        return "text_only_instructional"
    return "other_manual_failure"


def summarize(cohorts: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    if not cohorts:
        raise ValueError("at least one cohort is required")
    all_rows: list[dict[str, str]] = []
    reports: dict[str, Any] = {}
    for name, rows in cohorts.items():
        if not rows:
            raise ValueError(f"empty cohort: {name}")
        item_ids = [row.get("item_id", "") for row in rows]
        uids = [row.get("uid", "") for row in rows]
        if not all(item_ids) or len(item_ids) != len(set(item_ids)):
            raise ValueError(f"{name}: empty or duplicate item_id")
        if not all(uids) or len(uids) != len(set(uids)):
            raise ValueError(f"{name}: cohort is not source-disjoint")
        routes = Counter(manual_route(row) for row in rows)
        salvage_forms = Counter(
            row.get("visual_form") or "unknown"
            for row in rows
            if manual_route(row) in {
                "instructional_demo_candidate",
                "instructional_demo_candidate_after_relabel",
            }
        )
        reports[name] = {
            "items": len(rows),
            "unique_sources": len(set(uids)),
            "manual_review_complete": all(
                row.get(field) in {"yes", "no"}
                for row in rows for field in FIELDS
            ),
            "routes": dict(sorted(routes.items())),
            "salvageable_demo_media": dict(sorted(salvage_forms.items())),
        }
        all_rows.extend(rows)
    uid_counts = Counter(row["uid"] for row in all_rows)
    aggregate_routes = Counter(manual_route(row) for row in all_rows)
    return {
        "kind": "instructional_manual_contract_route_audit_v1",
        "policy": "manual_evidence_only_non_destructive_review_routing",
        "items": len(all_rows),
        "unique_sources": len(uid_counts),
        "cross_cohort_overlapping_sources": sum(
            count > 1 for count in uid_counts.values()
        ),
        "cohort_rates_must_be_reported_separately": True,
        "aggregate_raw_route_counts_not_an_independent_prevalence_estimate": dict(
            sorted(aggregate_routes.items())
        ),
        "cohorts": reports,
        "automatic_acceptance": False,
        "corpus_mutated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", action="append", nargs=2, metavar=("NAME", "LEDGER"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("refusing to overwrite existing audit artifact")
    report = summarize({name: read_tsv(Path(path)) for name, path in args.cohort})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

