#!/usr/bin/env python3
"""Evaluate literal instructional query provenance as manual review priority."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def yes(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "y", "true", "1"}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if not n:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def load_cohort(spec: dict[str, Any], root: Path, selected_value: str) -> list[dict[str, Any]]:
    manual_path = root / spec["manual"]
    metadata_path = root / spec["metadata"]
    for path, expected in (
        (manual_path, spec["manual_sha256"]),
        (metadata_path, spec["metadata_sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    with manual_path.open() as handle:
        manual = {row["item_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    metadata = read_jsonl(metadata_path)
    if len(metadata) != spec["items"] or len({row["uid"] for row in metadata}) != len(metadata):
        raise ValueError(f"cohort size or source disjointness changed: {spec['name']}")
    if set(manual) != {str(row["item_id"]) for row in metadata}:
        raise ValueError(f"manual ledger does not exactly cover metadata: {spec['name']}")
    rows = []
    for item in metadata:
        label = manual[str(item["item_id"])]
        query_source = str(item.get("query_source") or "")
        rows.append({
            "cohort": spec["name"],
            "candidate_id": str(item["candidate_id"]),
            "item_id": str(item["item_id"]),
            "uid": str(item["uid"]),
            "query_source": query_source,
            "retro_priority": query_source == selected_value,
            "manual_visual_demo": yes(label["visual_demo"]),
            "manual_usable_demo": yes(label["usable_demo"]),
            "visual_form": str(label.get("visual_form") or ""),
            "manual_note": str(label.get("note") or ""),
        })
    return rows


def binary_metrics(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    selected = [row for row in rows if row["retro_priority"]]
    controls = [row for row in rows if not row["retro_priority"]]
    positives = [row for row in rows if row[target]]
    tp = sum(row[target] for row in selected)
    fp = len(selected) - tp
    fn = len(positives) - tp
    control_positive = sum(row[target] for row in controls)
    precision = tp / len(selected) if selected else None
    control_rate = control_positive / len(controls) if controls else None
    return {
        "n": len(rows),
        "selected": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "precision_wilson_95": wilson(tp, len(selected)),
        "recall": tp / len(positives) if positives else None,
        "control_n": len(controls),
        "control_positive": control_positive,
        "control_rate": control_rate,
        "precision_delta_over_control": (
            precision - control_rate
            if precision is not None and control_rate is not None else None
        ),
    }


def evaluate(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    by_cohort = {}
    for name in [spec["name"] for spec in contract["cohorts"]]:
        cohort = [row for row in rows if row["cohort"] == name]
        by_cohort[name] = {
            "visual_demo": binary_metrics(cohort, "manual_visual_demo"),
            "usable_demo": binary_metrics(cohort, "manual_usable_demo"),
        }
    uids = {name: {row["uid"] for row in rows if row["cohort"] == name} for name in by_cohort}
    names = list(uids)
    overlap = len(uids[names[0]] & uids[names[1]])
    gate = contract["gate"]
    checks = {
        "source_uid_overlap": overlap == gate["source_uid_overlap"],
        "minimum_total_selected": sum(
            value["visual_demo"]["selected"] for value in by_cohort.values()
        ) >= gate["minimum_total_selected"],
        "minimum_selected_visual_precision_each_cohort": all(
            value["visual_demo"]["precision"] is not None
            and value["visual_demo"]["precision"]
            >= gate["minimum_selected_visual_precision_each_cohort"]
            for value in by_cohort.values()
        ),
        "minimum_visual_precision_delta_each_cohort": all(
            value["visual_demo"]["precision_delta_over_control"] is not None
            and value["visual_demo"]["precision_delta_over_control"]
            >= gate["minimum_visual_precision_delta_each_cohort"]
            for value in by_cohort.values()
        ),
    }
    passed = all(checks.values())
    return {
        "kind": "instructional_retro_query_source_manual_replication_v1",
        "status": contract["status"],
        "policy": contract["policy"],
        "items": len(rows),
        "manual_review_complete": True,
        "source_uid_overlap": overlap,
        "cohorts": by_cohort,
        "aggregate": {
            "visual_demo": binary_metrics(rows, "manual_visual_demo"),
            "usable_demo": binary_metrics(rows, "manual_usable_demo"),
        },
        "gate_checks": checks,
        "review_ranking_gate_passed": passed,
        "allowed_use": "manual_review_priority_only" if passed else "untrusted_diagnostic_only",
        "automatic_acceptance": False,
        "corpus_mutated": False,
    }


def run(contract_path: Path, output: Path, records_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    root = contract_path.resolve().parents[2]
    rows = []
    for spec in contract["cohorts"]:
        rows.extend(load_cohort(spec, root, contract["signal"]["selected_value"]))
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = evaluate(rows, contract)
    report["artifact_sha256"] = {
        "analysis_contract": sha256(contract_path),
        "normalized_records": sha256(records_path),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.contract, args.out, args.records), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
