#!/usr/bin/env python3
"""Validate a frozen post-reveal witnessed audit without changing the corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRI = {"yes", "no", "uncertain"}
ROUTES = {
    "witnessed",
    "instructional",
    "commentary",
    "negative",
    "technical",
    "unusable",
    "uncertain",
}
EVIDENCE_FIELDS = (
    "literal_social_norm_violation_visible",
    "assigned_norm_matches_event",
    "reaction_is_organic_bystander_disapproval",
    "violation_precedes_reaction",
    "reaction_spliceable_without_label_leak",
)
REQUIRED_FIELDS = {
    "item_id",
    "audit_index",
    *EVIDENCE_FIELDS,
    "strict_witnessed_pass",
    "recoverable_route",
    "required_repair",
    "evidence_note",
    "review_complete",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def validate(
    packets_path: Path,
    sparse_path: Path,
    review_path: Path,
) -> dict[str, Any]:
    packets = read_jsonl(packets_path)
    sparse = read_jsonl(sparse_path)
    reviews = read_jsonl(review_path)
    expected_ids = [str(row.get("item_id")) for row in packets]
    review_ids = [str(row.get("item_id")) for row in reviews]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("semantic packets contain duplicate item_id values")
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("review contains duplicate item_id values")
    if review_ids != expected_ids:
        raise ValueError("review item order/coverage does not exactly match packets")
    sparse_ids = {str(row.get("item_id")) for row in sparse}
    if sparse_ids != set(expected_ids):
        raise ValueError("sparse review coverage does not match semantic packets")
    packet_index = {
        str(row["item_id"]): row.get("audit_index")
        for row in packets
    }
    strict_passes = 0
    routes: dict[str, int] = {}
    for row in reviews:
        item_id = str(row["item_id"])
        if set(row) != REQUIRED_FIELDS:
            raise ValueError(f"{item_id}: schema mismatch")
        if row["audit_index"] != packet_index[item_id]:
            raise ValueError(f"{item_id}: audit_index mismatch")
        for field in EVIDENCE_FIELDS:
            if row[field] not in TRI:
                raise ValueError(f"{item_id}: invalid {field}")
        if row["strict_witnessed_pass"] not in {"yes", "no"}:
            raise ValueError(f"{item_id}: invalid strict_witnessed_pass")
        if row["recoverable_route"] not in ROUTES:
            raise ValueError(f"{item_id}: invalid recoverable_route")
        if row["review_complete"] != "yes":
            raise ValueError(f"{item_id}: review is incomplete")
        if not str(row["evidence_note"]).strip():
            raise ValueError(f"{item_id}: missing evidence_note")
        if not str(row["required_repair"]).strip():
            raise ValueError(f"{item_id}: missing required_repair")
        all_yes = all(row[field] == "yes" for field in EVIDENCE_FIELDS)
        expected_pass = all_yes and row["recoverable_route"] == "witnessed"
        if (row["strict_witnessed_pass"] == "yes") != expected_pass:
            raise ValueError(f"{item_id}: strict pass is inconsistent with evidence")
        strict_passes += int(row["strict_witnessed_pass"] == "yes")
        route = str(row["recoverable_route"])
        routes[route] = routes.get(route, 0) + 1
    return {
        "coverage_exact": True,
        "items": len(reviews),
        "strict_witnessed_passes": strict_passes,
        "strict_witnessed_rate": strict_passes / len(reviews) if reviews else None,
        "routes": dict(sorted(routes.items())),
        "policy": "manual_post_reveal_no_corpus_mutation",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--sparse-review", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.packets, args.sparse_review, args.review),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
