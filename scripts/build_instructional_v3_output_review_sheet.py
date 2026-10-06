#!/usr/bin/env python3
"""Build a fail-closed row-level manual review sheet for V3 outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_instructional_demo_adjudicator_v3 import (
        audit_index,
        binary_label,
        read_tsv,
    )
    from scripts.evaluate_instructional_demo_consensus_v2 import latest_success, read_jsonl
except ModuleNotFoundError:
    from evaluate_instructional_demo_adjudicator_v3 import (  # type: ignore[no-redef]
        audit_index,
        binary_label,
        read_tsv,
    )
    from evaluate_instructional_demo_consensus_v2 import (  # type: ignore[no-redef]
        latest_success,
        read_jsonl,
    )


FIELDS = [
    "audit_index",
    "item_id",
    "manual_visual_demo",
    "manual_visual_form",
    "manual_visual_evidence",
    "v3_demo_candidate",
    "v3_visual_sequence",
    "v3_participant_configuration",
    "v3_behavior_evidence",
    "v3_specific_social_act",
    "v3_literal_behavior",
    "v3_evidence",
    "output_reviewed",
    "output_visually_supported",
    "error_mechanism",
    "note",
]


def build_rows(
    manifest: list[dict[str, Any]],
    manual: list[dict[str, str]],
    outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_index = {audit_index(row): row for row in manifest}
    gold = {int(row["audit_index"]): row for row in manual}
    model = latest_success(outputs)
    if not set(by_index).issubset(gold):
        raise ValueError("manual ledger does not cover model manifest")
    if {str(row["item_id"]) for row in manifest} != set(model):
        raise ValueError("manifest/model successful coverage mismatch")
    rows: list[dict[str, Any]] = []
    for index in sorted(by_index):
        source = by_index[index]
        item_id = str(source["item_id"])
        manual_row = gold[index]
        result = model[item_id]["result"]
        rows.append({
            "audit_index": index,
            "item_id": item_id,
            "manual_visual_demo": "yes" if binary_label(manual_row) else "no",
            "manual_visual_form": manual_row.get("visual_form", ""),
            "manual_visual_evidence": manual_row.get("manual_evidence", ""),
            "v3_demo_candidate": "yes" if result["demo_candidate"] else "no",
            "v3_visual_sequence": result["visual_sequence"],
            "v3_participant_configuration": result["participant_configuration"],
            "v3_behavior_evidence": result["behavior_evidence"],
            "v3_specific_social_act": result["specific_social_act"],
            "v3_literal_behavior": result["literal_behavior"],
            "v3_evidence": result["evidence"],
            "output_reviewed": "",
            "output_visually_supported": "",
            "error_mechanism": "",
            "note": "",
        })
    return rows


def apply_review_overlay(
    rows: list[dict[str, Any]], overlay: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Apply a complete human-authored review overlay, failing closed."""
    by_index = {int(row["audit_index"]): row for row in overlay}
    expected = {int(row["audit_index"]) for row in rows}
    if set(by_index) != expected:
        raise ValueError("review overlay coverage mismatch")
    allowed = {"yes", "no"}
    merged = []
    for row in rows:
        review = by_index[int(row["audit_index"])]
        reviewed = review.get("output_reviewed", "").strip().lower()
        supported = review.get("output_visually_supported", "").strip().lower()
        if reviewed != "yes":
            raise ValueError(f"audit_index {row['audit_index']}: output not reviewed")
        if supported not in allowed:
            raise ValueError(
                f"audit_index {row['audit_index']}: invalid visual-support label"
            )
        merged.append({
            **row,
            "output_reviewed": reviewed,
            "output_visually_supported": supported,
            "error_mechanism": review.get("error_mechanism", "").strip(),
            "note": review.get("note", "").strip(),
        })
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manual-ledger", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--review-overlay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(
        read_jsonl(args.manifest), read_tsv(args.manual_ledger), read_jsonl(args.outputs)
    )
    if args.review_overlay is not None:
        rows = apply_review_overlay(rows, read_tsv(args.review_overlay))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"items": len(rows), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
