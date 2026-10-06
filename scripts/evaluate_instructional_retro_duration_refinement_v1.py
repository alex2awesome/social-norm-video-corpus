#!/usr/bin/env python3
"""Evaluate a post-hoc duration refinement without authorizing its use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def yes(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "y", "true", "1"}


def load_cohort(spec: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    metadata_path, manual_path = root / spec["metadata"], root / spec["manual"]
    for path, expected in (
        (metadata_path, spec["metadata_sha256"]),
        (manual_path, spec["manual_sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    metadata = read_jsonl(metadata_path)
    with manual_path.open(newline="") as handle:
        manual = {row["item_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    if len(metadata) != int(spec["items"]):
        raise ValueError(f"cohort size changed: {spec['name']}")
    if len({row["uid"] for row in metadata}) != len(metadata):
        raise ValueError(f"cohort is not source-disjoint: {spec['name']}")
    if set(manual) != {str(row["item_id"]) for row in metadata}:
        raise ValueError(f"manual ledger does not exactly cover metadata: {spec['name']}")
    return [{
        "cohort": spec["name"],
        "item_id": str(row["item_id"]),
        "uid": str(row["uid"]),
        "query_source": str(row.get("query_source") or ""),
        "duration_seconds": float(row["duration_hint"]),
        "manual_visual_demo": yes(manual[str(row["item_id"])]["visual_demo"]),
        "manual_usable_demo": yes(manual[str(row["item_id"])]["usable_demo"]),
    } for row in metadata]


def rate(
    rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool], target: str
) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    positives = sum(bool(row[target]) for row in selected)
    return {
        "selected": len(selected),
        "positive": positives,
        "rate": positives / len(selected) if selected else None,
    }


def evaluate(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["signal"]["query_source"]
    minimum = float(contract["signal"]["minimum_duration_seconds"])
    retro = lambda row: row["query_source"] == source
    refined = lambda row: retro(row) and row["duration_seconds"] >= minimum
    cohorts = {}
    for name in [spec["name"] for spec in contract["cohorts"]]:
        cohort = [row for row in rows if row["cohort"] == name]
        cohorts[name] = {
            target: {
                "retro_only": rate(cohort, retro, target),
                "retro_and_min_duration": rate(cohort, refined, target),
            }
            for target in ("manual_visual_demo", "manual_usable_demo")
        }
    return {
        "kind": "instructional_retro_duration_refinement_discovery_v1",
        "status": contract["status"],
        "items": len(rows),
        "manual_review_complete": True,
        "cohorts": cohorts,
        "aggregate": {
            target: {
                "retro_only": rate(rows, retro, target),
                "retro_and_min_duration": rate(rows, refined, target),
            }
            for target in ("manual_visual_demo", "manual_usable_demo")
        },
        "future_replication_required": True,
        "operational_rule_promoted": False,
        "automatic_acceptance": False,
        "corpus_mutated": False,
    }


def run(contract_path: Path, records_path: Path, output_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    root = contract_path.resolve().parents[2]
    rows = []
    for spec in contract["cohorts"]:
        rows.extend(load_cohort(spec, root))
    records_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = evaluate(rows, contract)
    report["artifact_sha256"] = {
        "analysis_contract": sha256(contract_path),
        "normalized_records": sha256(records_path),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.records.exists() or args.out.exists():
        raise SystemExit("refusing to overwrite evaluation artifacts")
    print(json.dumps(run(args.contract, args.records, args.out), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
