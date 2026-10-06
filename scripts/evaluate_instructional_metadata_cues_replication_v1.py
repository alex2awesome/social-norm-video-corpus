#!/usr/bin/env python3
"""Require instructional metadata cues to replicate across two manual cohorts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_instructional_metadata_cues_v1 import cue_values, metric, read_jsonl
except ModuleNotFoundError:
    from evaluate_instructional_metadata_cues_v1 import cue_values, metric, read_jsonl  # type: ignore[no-redef]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "y", "true", "1"}


def load_development(labels_path: Path, metadata_path: Path) -> list[dict[str, Any]]:
    with labels_path.open() as handle:
        labels = list(csv.DictReader(handle, delimiter="\t"))
    metadata = {str(row["item_id"]): row for row in read_jsonl(metadata_path)}
    rows = []
    for label in labels:
        item_id = str(label["item_id"])
        if item_id not in metadata:
            raise ValueError(f"missing development metadata for {item_id}")
        raw = metadata[item_id]
        rows.append(
            {
                "cohort": "development_100",
                "candidate_id": str(label["candidate_id"]),
                "item_id": item_id,
                "uid": str(label["uid"]),
                "manual_visual_demo": as_bool(label["visual_demo"]),
                "manual_exact_original": as_bool(label["usable_demo"]),
                "visual_form": str(label.get("visual_form") or ""),
                "manual_note": str(label.get("note") or ""),
                "cues": cue_values(raw),
            }
        )
    return rows


def load_evaluation(path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in read_jsonl(path):
        rows.append(
            {
                "cohort": "evaluation_60",
                "candidate_id": str(raw["candidate_id"]),
                "item_id": str(raw["item_id"]),
                "uid": str(raw["uid"]),
                "manual_visual_demo": bool(raw["manual_visual_demo"]),
                "manual_exact_original": bool(raw["manual_exact_original"]),
                "visual_form": str(raw.get("visual_form") or ""),
                "manual_note": str(raw.get("blind_visual_note") or ""),
                "cues": cue_values(raw),
            }
        )
    return rows


def cohort_pass(result: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        result["selected"] >= gate["minimum_selected"]
        and result["precision"] is not None
        and result["precision"] >= gate["ranking_gate_visual_precision"]
        and result["precision_delta_over_base"] >= gate["ranking_gate_visual_precision_delta_over_base"]
        and result["recall"] >= gate["ranking_gate_visual_recall"]
    )


def evaluate(
    development: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    parent: dict[str, Any],
    amendment: dict[str, Any],
) -> dict[str, Any]:
    overlap = {row["uid"] for row in development} & {row["uid"] for row in evaluation}
    if overlap:
        raise ValueError(f"source overlap between cohorts: {sorted(overlap)}")
    cues = list(parent["cues"])
    cohorts = {"development_100": development, "evaluation_60": evaluation}
    metrics = {
        cue: {
            cohort: {
                "visual_demo": metric(rows, cue, "manual_visual_demo"),
                "exact_demo": metric(rows, cue, "manual_exact_original"),
            }
            for cohort, rows in cohorts.items()
        }
        for cue in cues
    }
    parent_gate = parent["evaluation"]
    replication_gate = amendment["replication_gate"]
    decisions = {}
    for cue in cues:
        passes = {
            cohort: cohort_pass(values["visual_demo"], parent_gate)
            for cohort, values in metrics[cue].items()
        }
        total_selected = sum(
            values["visual_demo"]["selected"] for values in metrics[cue].values()
        )
        replicated = (
            all(passes.values())
            and total_selected >= replication_gate["minimum_total_selected"]
        )
        decisions[cue] = {
            "cohort_gate_passed": passes,
            "total_selected": total_selected,
            "replicated_ranking_gate_passed": replicated,
            "allowed_use": "manual_review_priority_only" if replicated else "untrusted_diagnostic_only",
            "automatic_acceptance": False,
        }
    return {
        "kind": "instructional_metadata_script_cues_v1_replication_evaluation",
        "policy": parent["policy"],
        "cohort_items": {name: len(rows) for name, rows in cohorts.items()},
        "source_overlap": 0,
        "metrics": metrics,
        "decisions": decisions,
        "all_outputs_manually_scored": sum(len(rows) for rows in cohorts.values()),
        "automatic_acceptance": False,
        "corpus_mutated": False,
    }


def run(
    development_labels: Path,
    development_metadata: Path,
    evaluation_records: Path,
    parent_prereg: Path,
    amendment_path: Path,
    output: Path,
    summary_path: Path,
) -> dict[str, Any]:
    parent = json.loads(parent_prereg.read_text())
    amendment = json.loads(amendment_path.read_text())
    expected = {
        development_labels: parent["development_cohort"]["labels_sha256"],
        development_metadata: amendment["development_metadata"]["records_sha256"],
        evaluation_records: parent["evaluation_cohort"]["records_sha256"],
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"input hash mismatch for {path}")
    development = load_development(development_labels, development_metadata)
    evaluation = load_evaluation(evaluation_records)
    all_rows = development + evaluation
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows))
    summary = evaluate(development, evaluation, parent, amendment)
    summary["artifact_sha256"] = {
        "parent_preregistration": sha256(parent_prereg),
        "replication_gate_amendment": sha256(amendment_path),
        "normalized_records": sha256(output),
        "development_labels": sha256(development_labels),
        "development_metadata": sha256(development_metadata),
        "evaluation_records": sha256(evaluation_records),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-labels", type=Path, required=True)
    parser.add_argument("--development-metadata", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--parent-preregistration", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.development_labels,
        args.development_metadata,
        args.evaluation_records,
        args.parent_preregistration,
        args.amendment,
        args.output,
        args.summary,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
