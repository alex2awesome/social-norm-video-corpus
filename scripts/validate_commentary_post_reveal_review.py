#!/usr/bin/env python3
"""Validate a frozen post-reveal commentary audit without corpus mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRI = {"yes", "no", "uncertain"}
ROUTES = {
    "commentary_visual",
    "commentary_text_only",
    "instructional_visual",
    "witnessed",
    "technical_or_formal",
    "unusable",
    "uncertain",
}
EVIDENCE_FIELDS = (
    "visual_event_present",
    "assigned_norm_is_social_norm",
    "statement_is_normative_evidence",
    "quote_grounded_in_transcript",
    "visible_event_matches_statement",
    "polarity_matches_visible_event",
    "visual_window_is_event_dominant",
)
REQUIRED_FIELDS = {
    "item_id",
    "audit_index",
    *EVIDENCE_FIELDS,
    "strict_commentary_visual_pass",
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


def keyed(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{source} contains a missing item_id")
        if item_id in result:
            raise ValueError(f"{source} contains duplicate item_id values")
        result[item_id] = row
    return result


def final_visual_event(
    item_id: str,
    sparse: dict[str, Any],
    dense_by_id: dict[str, dict[str, Any]],
) -> str:
    # Statement-centered commentary audits directly record whether a local
    # human event and a social interaction are visible. Preserve support for
    # the older scene/action decomposition below.
    if "localized_human_event_visible" in sparse:
        if item_id in dense_by_id:
            raise ValueError(f"{item_id}: unexpected dense review")
        human = sparse.get("localized_human_event_visible")
        social = sparse.get("localized_social_interaction_visible")
        if human not in TRI or social not in TRI:
            raise ValueError(f"{item_id}: invalid frozen visual judgment")
        if human == "yes" and social == "yes":
            return "yes"
        if human == "no" or social == "no":
            return "no"
        return "uncertain"

    requires_dense = sparse.get("dense_review_required") == "yes"
    if requires_dense != (item_id in dense_by_id):
        raise ValueError(f"{item_id}: dense review coverage is inconsistent")
    if requires_dense:
        dense = dense_by_id[item_id]
        if dense.get("review_complete") != "yes":
            raise ValueError(f"{item_id}: dense review is incomplete")
        scene = dense.get("dense_scene_visible")
        action = dense.get("dense_concrete_action_visible")
    else:
        scene = sparse.get("situated_social_scene")
        action = sparse.get("concrete_behavior_visible")
    if scene not in TRI or action not in TRI:
        raise ValueError(f"{item_id}: invalid frozen visual judgment")
    if scene == "yes" and action == "yes":
        return "yes"
    if scene == "no" or action == "no":
        return "no"
    return "uncertain"


def validate(
    packets_path: Path,
    sparse_path: Path,
    dense_path: Path,
    review_path: Path,
) -> dict[str, Any]:
    packets = read_jsonl(packets_path)
    sparse_rows = read_jsonl(sparse_path)
    dense_rows = read_jsonl(dense_path)
    reviews = read_jsonl(review_path)
    expected_ids = [str(row.get("item_id")) for row in packets]
    review_ids = [str(row.get("item_id")) for row in reviews]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("semantic packets contain duplicate item_id values")
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("review contains duplicate item_id values")
    if review_ids != expected_ids:
        raise ValueError("review item order/coverage does not exactly match packets")
    sparse_by_id = keyed(sparse_rows, "sparse review")
    dense_by_id = keyed(dense_rows, "dense review")
    if set(sparse_by_id) != set(expected_ids):
        raise ValueError("sparse review coverage does not match semantic packets")
    packet_index = {
        str(row["item_id"]): row.get("audit_index")
        for row in packets
    }

    strict_passes = 0
    visual_events = {"yes": 0, "no": 0, "uncertain": 0}
    routes: dict[str, int] = {}
    failure_reasons = {field: 0 for field in EVIDENCE_FIELDS}
    for row in reviews:
        item_id = str(row["item_id"])
        if set(row) != REQUIRED_FIELDS:
            raise ValueError(f"{item_id}: schema mismatch")
        if row["audit_index"] != packet_index[item_id]:
            raise ValueError(f"{item_id}: audit_index mismatch")
        for field in EVIDENCE_FIELDS:
            if row[field] not in TRI:
                raise ValueError(f"{item_id}: invalid {field}")
        expected_visual = final_visual_event(
            item_id,
            sparse_by_id[item_id],
            dense_by_id,
        )
        if row["visual_event_present"] != expected_visual:
            raise ValueError(
                f"{item_id}: visual_event_present contradicts frozen blind review"
            )
        if (
            row["visual_event_present"] != "yes"
            and row["visible_event_matches_statement"] == "yes"
        ):
            raise ValueError(f"{item_id}: absent event cannot visibly match statement")
        if (
            row["visual_event_present"] != "yes"
            and row["visual_window_is_event_dominant"] == "yes"
        ):
            raise ValueError(f"{item_id}: absent event cannot dominate visual window")
        if row["strict_commentary_visual_pass"] not in {"yes", "no"}:
            raise ValueError(f"{item_id}: invalid strict_commentary_visual_pass")
        if row["recoverable_route"] not in ROUTES:
            raise ValueError(f"{item_id}: invalid recoverable_route")
        if row["review_complete"] != "yes":
            raise ValueError(f"{item_id}: review is incomplete")
        if not str(row["evidence_note"]).strip():
            raise ValueError(f"{item_id}: missing evidence_note")
        if not str(row["required_repair"]).strip():
            raise ValueError(f"{item_id}: missing required_repair")
        all_yes = all(row[field] == "yes" for field in EVIDENCE_FIELDS)
        expected_pass = all_yes and row["recoverable_route"] == "commentary_visual"
        if (row["strict_commentary_visual_pass"] == "yes") != expected_pass:
            raise ValueError(f"{item_id}: strict pass is inconsistent with evidence")
        strict_passes += int(row["strict_commentary_visual_pass"] == "yes")
        visual_events[row["visual_event_present"]] += 1
        route = str(row["recoverable_route"])
        routes[route] = routes.get(route, 0) + 1
        for field in EVIDENCE_FIELDS:
            failure_reasons[field] += int(row[field] != "yes")
    return {
        "coverage_exact": True,
        "items": len(reviews),
        "visual_events": visual_events,
        "strict_commentary_visual_passes": strict_passes,
        "strict_commentary_visual_rate": (
            strict_passes / len(reviews) if reviews else None
        ),
        "routes": dict(sorted(routes.items())),
        "non_yes_by_gate": failure_reasons,
        "policy": "manual_post_reveal_no_corpus_mutation",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--sparse-review", required=True, type=Path)
    parser.add_argument("--dense-review", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                args.packets,
                args.sparse_review,
                args.dense_review,
                args.review,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
