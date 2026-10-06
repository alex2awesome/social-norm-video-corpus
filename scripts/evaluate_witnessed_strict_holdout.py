#!/usr/bin/env python3
"""Freeze strict witnessed VLM replication against a manual holdout.

This evaluator understands append-only model logs: failed attempts remain in
the source file, while the latest successful result for each item is used.
Every item selected by either model is joined back to the frozen manual
adjudication so that a promotion decision cannot be made from aggregate metrics
alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_witnessed_atomic_benchmark import witnessed_atomic
else:
    from evaluate_witnessed_atomic_benchmark import witnessed_atomic


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collapse_successful(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], int]:
    """Return latest successful record by item plus failed-attempt count."""
    successful: dict[str, dict[str, Any]] = {}
    failures = 0
    for row in rows:
        if row.get("error") is None and isinstance(row.get("result"), dict):
            successful[row["item_id"]] = row
        else:
            failures += 1
    return successful, failures


def selected(record: dict[str, Any], band: str) -> bool:
    result = record["result"]
    if band == "review_triage":
        return witnessed_atomic(result, require_bounds=False)
    if band == "recovery_exact_bounds":
        return witnessed_atomic(result, require_bounds=True)
    if band == "strict_current_exact_bounds":
        return (
            witnessed_atomic(result, require_bounds=True)
            and result.get("proposed_label_relation") == "exact"
        )
    raise ValueError(f"unknown band: {band}")


def binary_metrics(
    item_ids: set[str],
    gold_positive: set[str],
    predicted_positive: set[str],
) -> dict[str, Any]:
    tp = predicted_positive & gold_positive
    fp = predicted_positive - gold_positive
    fn = gold_positive - predicted_positive
    tn = item_ids - predicted_positive - gold_positive
    return {
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "tn": len(tn),
        "precision": len(tp) / len(predicted_positive)
        if predicted_positive
        else None,
        "recall": len(tp) / len(gold_positive) if gold_positive else None,
        "predicted_positive_count": len(predicted_positive),
        "predicted_positive_items": sorted(predicted_positive),
        "true_positive_items": sorted(tp),
        "false_positive_items": sorted(fp),
        "false_negative_items": sorted(fn),
    }


def repair_counts(records: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records.values():
        result = record["result"]
        for key in (
            "temporal_bounds_validation_repair",
            "authenticity_validation_repair",
        ):
            if key in result:
                value = str(result[key])
                counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_report(
    root: Path,
    model_paths: dict[str, Path],
) -> dict[str, Any]:
    manifest_rows = load_jsonl(root / "vlm_manifest.jsonl")
    manifest = {row["item_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("holdout manifest contains duplicate item IDs")
    item_ids = set(manifest)
    gold_positive = {
        row["item_id"] for row in manifest_rows if row.get("gold_usable") is True
    }

    blind_rows = load_jsonl(root / "manual_blind_witnessed_review.jsonl")
    blind = {int(row["audit_index"]): row for row in blind_rows}
    adjudication = json.loads(
        (root / "manual_revealed_label_adjudication.json").read_text()
    )
    strict = adjudication["strict_witnessed_contract"]
    accepted = set(map(int, strict["accepted_indices"]))
    rejected = set(map(int, strict["rejected_indices"]))
    unresolved = set(
        map(int, strict["source_authenticity_unresolved_indices"])
    )
    expected_indices = {int(row["audit_index"]) for row in manifest_rows}
    if accepted | rejected | unresolved != expected_indices:
        raise ValueError("manual adjudication does not cover the full holdout")
    if (accepted & rejected) or (accepted & unresolved) or (rejected & unresolved):
        raise ValueError("manual adjudication dispositions overlap")
    if {manifest_rows[index]["audit_index"] for index in range(len(manifest_rows))} != expected_indices:
        raise ValueError("manifest audit indices are not unique")

    failure_groups_by_index: dict[int, list[str]] = {}
    for group, indices in adjudication["failure_groups"].items():
        for index in indices:
            failure_groups_by_index.setdefault(int(index), []).append(group)

    models: dict[str, dict[str, dict[str, Any]]] = {}
    model_metadata: dict[str, Any] = {}
    for name, path in model_paths.items():
        records, failures = collapse_successful(load_jsonl(path))
        if set(records) != item_ids:
            missing = sorted(item_ids - set(records))
            extra = sorted(set(records) - item_ids)
            raise ValueError(
                f"{name} does not cover holdout: missing={missing}, extra={extra}"
            )
        models[name] = records
        model_metadata[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "successful_unique_items": len(records),
            "failed_append_only_attempts": failures,
            "fail_closed_validation_repairs": repair_counts(records),
        }

    bands = (
        "review_triage",
        "recovery_exact_bounds",
        "strict_current_exact_bounds",
    )
    band_reports: dict[str, Any] = {}
    selected_sets: dict[tuple[str, str], set[str]] = {}
    for band in bands:
        for model_name, records in models.items():
            selected_sets[(band, model_name)] = {
                item_id
                for item_id, record in records.items()
                if selected(record, band)
            }
        per_model = {
            name: binary_metrics(
                item_ids,
                gold_positive,
                selected_sets[(band, name)],
            )
            for name in models
        }
        selection_values = list(
            selected_sets[(band, name)] for name in models
        )
        intersection = set.intersection(*selection_values)
        union = set.union(*selection_values)
        band_reports[band] = {
            "models": per_model,
            "intersection": binary_metrics(
                item_ids, gold_positive, intersection
            ),
            "union": binary_metrics(item_ids, gold_positive, union),
        }

    all_selected = set().union(
        *(
            selected_sets[(band, name)]
            for band in bands
            for name in models
        )
    )
    manual_selected_audit = []
    for item_id in sorted(
        all_selected, key=lambda value: int(manifest[value]["audit_index"])
    ):
        row = manifest[item_id]
        index = int(row["audit_index"])
        if index in accepted:
            disposition = "strict_accept"
        elif index in unresolved:
            disposition = "source_authenticity_unresolved_fail_closed"
        else:
            disposition = "strict_reject"
        manual_selected_audit.append(
            {
                "item_id": item_id,
                "audit_index": index,
                "uid": row["uid"],
                "manual_disposition": disposition,
                "blind_description": blind[index]["description"],
                "manual_failure_groups": failure_groups_by_index.get(index, []),
                "selected_by": {
                    band: [
                        name
                        for name in models
                        if item_id in selected_sets[(band, name)]
                    ]
                    for band in bands
                },
            }
        )

    strict_intersection = band_reports["strict_current_exact_bounds"][
        "intersection"
    ]
    return {
        "kind": "witnessed_strict_vlm_holdout_replication",
        "status": "failed_promotion_shadow_only",
        "holdout_items": len(item_ids),
        "manual_gold_positive_count": len(gold_positive),
        "manual_gold_positive_items": sorted(gold_positive),
        "manual_review_coverage": {
            "blind_reviewed": len(blind),
            "revealed_adjudicated": len(accepted | rejected | unresolved),
            "model_selected_union_reviewed": len(manual_selected_audit),
            "model_selected_union_total": len(all_selected),
            "complete": len(manual_selected_audit) == len(all_selected),
        },
        "model_outputs": model_metadata,
        "bands": band_reports,
        "manual_audit_of_every_selected_item": manual_selected_audit,
        "promotion_decision": {
            "promoted": False,
            "reason": (
                "The strict two-model intersection selected "
                f"{strict_intersection['predicted_positive_count']} items with "
                f"{strict_intersection['tp']} true positives and "
                f"{strict_intersection['fp']} false positives; it also missed "
                f"{strict_intersection['fn']} of {len(gold_positive)} manual "
                "positives."
            ),
            "production_changes": False,
            "destructive_actions": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--glm", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.root,
        {
            "glm_4_6v_flash": args.glm,
            "qwen3_vl_8b_instruct": args.qwen,
        },
    )
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["promotion_decision"], sort_keys=True))


if __name__ == "__main__":
    main()
