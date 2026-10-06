#!/usr/bin/env python3
"""Validate sealed lineage and source-disjointness of commentary V3 transfer selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.select_commentary_occurred_event_transfer_v3 import ACCEPTED, read_jsonl, sha256
else:
    from select_commentary_occurred_event_transfer_v3 import ACCEPTED, read_jsonl, sha256


def validate_selection(
    selected: list[dict[str, Any]],
    excluded_uids: set[str],
    accepted_count: int,
    rejected_count: int,
) -> dict[str, Any]:
    if len(selected) != accepted_count + rejected_count:
        raise ValueError("selected count differs from preregistered decision strata")
    ids = [str(row.get("item_id") or "") for row in selected]
    uids = [str(row.get("uid") or "") for row in selected]
    if any(not value for value in ids + uids):
        raise ValueError("selection has blank item_id or uid")
    if len(ids) != len(set(ids)) or len(uids) != len(set(uids)):
        raise ValueError("selection is not item- and source-disjoint")
    if set(uids) & excluded_uids:
        raise ValueError("selection overlaps the design cohort")
    if [row.get("transfer_index") for row in selected] != list(range(len(selected))):
        raise ValueError("transfer indices are missing or reordered")
    observed_accepts = sum(row.get("v1_decision") in ACCEPTED for row in selected)
    observed_rejects = sum(row.get("v1_decision") == "reject" for row in selected)
    if (observed_accepts, observed_rejects) != (accepted_count, rejected_count):
        raise ValueError("prior-decision balance differs from preregistration")

    cache: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in selected:
        manifest_path = Path(row["source_manifest_path"])
        results_path = Path(row["source_results_path"])
        if sha256(manifest_path) != row.get("source_manifest_sha256"):
            raise ValueError(f"{row['item_id']}: source manifest hash changed")
        if sha256(results_path) != row.get("source_results_sha256"):
            raise ValueError(f"{row['item_id']}: source results hash changed")
        key = (str(manifest_path), str(results_path))
        if key not in cache:
            manifest = json.loads(manifest_path.read_text())
            cache[key] = (
                {item["item_id"]: item for item in manifest.get("items") or []},
                {item["item_id"]: item for item in read_jsonl(results_path)},
            )
        manifest_items, result_items = cache[key]
        item_id = row["item_id"]
        if item_id not in manifest_items or item_id not in result_items:
            raise ValueError(f"{item_id}: missing from linked source files")
        item = manifest_items[item_id]
        result = result_items[item_id]
        if item.get("uid") != row["uid"] or result.get("decision") != row["v1_decision"]:
            raise ValueError(f"{item_id}: source decision or UID lineage changed")
        if result.get("normalized_behavior", "") != row.get("v1_normalized_behavior", ""):
            raise ValueError(f"{item_id}: behavior lineage changed")
        if result.get("normalized_norm", "") != row.get("v1_normalized_norm", ""):
            raise ValueError(f"{item_id}: norm lineage changed")
        content_hash = str(item.get("content_manifest_sha256") or "")
        if content_hash != row.get("content_manifest_sha256"):
            raise ValueError(f"{item_id}: content-manifest lineage changed")
        if row.get("manual_reaudit_required") is not True:
            raise ValueError(f"{item_id}: manual re-audit flag missing")

    return {
        "kind": "commentary_occurred_event_v3_transfer_selection_validation",
        "selected": len(selected),
        "prior_accepts": observed_accepts,
        "prior_rejects": observed_rejects,
        "source_disjoint": True,
        "design_cohort_disjoint": True,
        "hash_lineage_valid": True,
        "manual_reaudit_required": True,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--exclude-gold", type=Path, required=True)
    parser.add_argument("--accepted-count", type=int, default=40)
    parser.add_argument("--rejected-count", type=int, default=20)
    args = parser.parse_args()
    excluded_uids = {
        row["uid"] for row in read_jsonl(args.exclude_gold) if row.get("uid")
    }
    report = validate_selection(
        read_jsonl(args.selection), excluded_uids, args.accepted_count, args.rejected_count
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
