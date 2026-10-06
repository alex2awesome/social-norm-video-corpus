#!/usr/bin/env python3
"""Fail closed unless a dense blind review covers its frozen manifest exactly."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_VISIBILITY = {"yes", "no", "uncertain"}
ALLOWED_CHANGED = {"yes", "no"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    return rows


def validate_dense_review(
    manifest_path: Path, sparse_review_path: Path, review_path: Path
) -> dict[str, Any]:
    manifest_rows = read_jsonl(manifest_path)
    sparse_rows = read_jsonl(sparse_review_path)
    review_rows = read_jsonl(review_path)
    if not manifest_rows:
        raise ValueError("dense manifest is empty")

    manifest_ids = [row.get("item_id") for row in manifest_rows]
    review_ids = [row.get("item_id") for row in review_rows]
    if any(not isinstance(item_id, str) or not item_id for item_id in manifest_ids):
        raise ValueError("every manifest row must have a nonempty item_id")
    if any(not isinstance(item_id, str) or not item_id for item_id in review_ids):
        raise ValueError("every review row must have a nonempty item_id")

    duplicate_manifest = sorted(
        item_id for item_id, count in Counter(manifest_ids).items() if count != 1
    )
    duplicate_review = sorted(
        item_id for item_id, count in Counter(review_ids).items() if count != 1
    )
    if duplicate_manifest:
        raise ValueError(f"duplicate manifest item_id values: {duplicate_manifest}")
    if duplicate_review:
        raise ValueError(f"duplicate review item_id values: {duplicate_review}")

    manifest_set = set(manifest_ids)
    review_set = set(review_ids)
    missing = sorted(manifest_set - review_set)
    extra = sorted(review_set - manifest_set)
    if missing or extra:
        raise ValueError(f"review coverage mismatch; missing={missing}, extra={extra}")

    manifest_by_id = {row["item_id"]: row for row in manifest_rows}
    sparse_ids = [row.get("item_id") for row in sparse_rows]
    duplicate_sparse = sorted(
        item_id
        for item_id, count in Counter(sparse_ids).items()
        if isinstance(item_id, str) and count != 1
    )
    if duplicate_sparse:
        raise ValueError(f"duplicate sparse-review item_id values: {duplicate_sparse}")
    sparse_by_id = {
        row["item_id"]: row
        for row in sparse_rows
        if isinstance(row.get("item_id"), str) and row["item_id"]
    }
    missing_sparse = sorted(manifest_set - set(sparse_by_id))
    if missing_sparse:
        raise ValueError(f"dense manifest items missing from sparse review: {missing_sparse}")
    required_fields = {
        "item_id",
        "audit_index",
        "dense_scene_visible",
        "dense_concrete_action_visible",
        "temporal_description",
        "sparse_changed",
        "review_complete",
    }
    for row_number, row in enumerate(review_rows, 1):
        unknown = sorted(set(row) - required_fields)
        missing_fields = sorted(required_fields - set(row))
        if unknown or missing_fields:
            raise ValueError(
                f"review row {row_number}: schema mismatch; "
                f"missing={missing_fields}, unknown={unknown}"
            )
        item_id = row["item_id"]
        expected_index = manifest_by_id[item_id].get("audit_index")
        if row["audit_index"] != expected_index:
            raise ValueError(
                f"{item_id}: audit_index={row['audit_index']!r}, "
                f"expected {expected_index!r}"
            )
        for field in ("dense_scene_visible", "dense_concrete_action_visible"):
            if row[field] not in ALLOWED_VISIBILITY:
                raise ValueError(f"{item_id}: invalid {field}={row[field]!r}")
        description = row["temporal_description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{item_id}: temporal_description must be nonempty")
        if row["sparse_changed"] not in ALLOWED_CHANGED:
            raise ValueError(f"{item_id}: invalid sparse_changed={row['sparse_changed']!r}")
        sparse = sparse_by_id[item_id]
        if sparse.get("dense_review_required") != "yes":
            raise ValueError(f"{item_id}: sparse review did not require dense follow-up")
        sparse_scene = sparse.get("situated_social_scene")
        sparse_action = sparse.get("concrete_behavior_visible")
        if sparse_scene not in ALLOWED_VISIBILITY or sparse_action not in ALLOWED_VISIBILITY:
            raise ValueError(f"{item_id}: sparse visibility labels are invalid")
        expected_changed = (
            "yes"
            if (
                row["dense_scene_visible"] != sparse_scene
                or row["dense_concrete_action_visible"] != sparse_action
            )
            else "no"
        )
        if row["sparse_changed"] != expected_changed:
            raise ValueError(
                f"{item_id}: sparse_changed={row['sparse_changed']!r}, "
                f"expected {expected_changed!r}"
            )
        if row["review_complete"] != "yes":
            raise ValueError(f"{item_id}: review_complete must be 'yes'")

    return {
        "manifest_items": len(manifest_rows),
        "review_items": len(review_rows),
        "coverage_exact": True,
        "schema_valid": True,
        "sparse_crosscheck_valid": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sparse-review", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    args = parser.parse_args()
    result = validate_dense_review(args.manifest, args.sparse_review, args.review)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
