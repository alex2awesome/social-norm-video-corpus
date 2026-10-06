#!/usr/bin/env python3
"""Create exact-coverage manual review ledgers for fresh VLM outputs.

The ledgers expose the model prediction only after the independent visual gold
pass.  They force a judgment of both the structured decision and its rationale;
an unaudited output can never be counted as supported merely because it parsed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COMMON = (
    "audit_index",
    "item_id",
    "uid",
    "model",
    "structured_prediction_supported",
    "rationale_supported",
    "input_modality_claim_supported",
    "unsupported_claim_types",
    "manual_rationale",
)
INSTRUCTIONAL = COMMON + (
    "predicted_segment_bounds_supported",
    "model_demo_candidate",
)
WITNESSED = COMMON + (
    "prompt_version",
    "reaction_decision_supported",
    "role_binding_supported",
    "trigger_binding_supported",
    "staging_decision_supported",
)
COMMENTARY = COMMON + (
    "stage_a_literal_analysis_supported",
    "stage_b_alignment_supported_by_stage_a",
    "model_strict_pass",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index_exact(rows: list[dict[str, Any]], field: str, name: str) -> dict[str, dict[str, Any]]:
    output = {str(row.get(field) or ""): row for row in rows}
    if not all(output) or len(output) != len(rows):
        raise ValueError(f"{name} has missing or duplicate {field}")
    return output


def build(
    pillar: str,
    cohort_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if pillar == "instructional":
        key, fields = "item_id", INSTRUCTIONAL
    elif pillar == "witnessed":
        key, fields = "candidate_id", WITNESSED
    elif pillar == "commentary":
        key, fields = "candidate_id", COMMENTARY
    else:
        raise ValueError(f"unknown pillar: {pillar}")
    cohort = index_exact(cohort_rows, key, "cohort")
    outputs = index_exact(output_rows, key, "model outputs")
    if set(cohort) != set(outputs):
        raise ValueError("successful model outputs do not exactly cover cohort")
    if any(row.get("error") not in (None, "") for row in outputs.values()):
        raise ValueError("model outputs contain errors; retry before manual review")

    records = []
    for audit_index, item_id in enumerate(sorted(cohort)):
        source = cohort[item_id]
        result = outputs[item_id]
        source_uid = str(source.get("uid") or "")
        output_uid = str(result.get("uid") or "")
        if source_uid and output_uid and source_uid != output_uid:
            raise ValueError(f"{item_id}: model output uid lineage mismatch")
        record = {field: "" for field in fields}
        record.update({
            "audit_index": str(audit_index),
            "item_id": item_id,
            "uid": source_uid or output_uid,
            "model": str(result.get("model") or ""),
        })
        if not record["uid"] or not record["model"]:
            raise ValueError(f"{item_id}: missing uid or model")
        if pillar == "instructional":
            prediction = result.get("result") or {}
            record["model_demo_candidate"] = (
                "yes" if prediction.get("demo_candidate") is True else "no"
            )
        elif pillar == "witnessed":
            record["prompt_version"] = str(result.get("prompt_version") or "")
            if not record["prompt_version"]:
                raise ValueError(f"{item_id}: missing prompt_version")
        else:
            record["model_strict_pass"] = (
                "yes" if result.get("strict_pass") is True else "no"
            )
        records.append(record)
    return fields, records


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pillar", choices=("instructional", "witnessed", "commentary"), required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    fields, rows = build(args.pillar, read_jsonl(args.cohort), read_jsonl(args.outputs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out, fields, rows)
    print(json.dumps({
        "pillar": args.pillar,
        "manual_output_review_rows": len(rows),
        "coverage_required": 1.0,
        "automatic_acceptance": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
