#!/usr/bin/env python3
"""Evaluate localized candidate VLM outputs at the witnessed clip level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.evaluate_witnessed_reaction_external_holdout import (
        load_jsonl,
        manual_target,
        metric_record,
    )
else:
    from evaluate_witnessed_reaction_external_holdout import (
        load_jsonl,
        manual_target,
        metric_record,
    )


TARGETED = {
    "targeted_objection",
    "correction_or_sanction",
    "protective_intervention",
}


def candidate_positive(
    result: dict[str, Any],
    *,
    include_authority: bool,
    allow_unresolved_visual_order: bool = False,
) -> bool:
    roles = {"bystander", "organic_audience"}
    if include_authority:
        roles.add("authority_or_host")
    if "triggering_action_visible" in result:
        return (
            result.get("triggering_action_visible") == "yes"
            and result.get("visible_response_at_candidate_moment") == "yes"
            and (
                result.get("response_after_or_overlaps_action") == "yes"
                or allow_unresolved_visual_order
            )
            and result.get("visibly_distinct_third_person_responds") == "yes"
            and result.get("visible_response_targets_action") == "yes"
            and result.get("visual_responder_role") in roles
            and result.get("visible_response_content")
            in TARGETED | {"interposition_or_separation"}
        )
    return (
        result.get("candidate_grounded") == "yes"
        and result.get("triggering_action_visible_or_audible") == "yes"
        and result.get("action_before_or_overlaps_candidate") == "yes"
        and result.get("reaction_targets_action") == "yes"
        and result.get("reactor_visibly_distinct_from_actor") == "yes"
        and result.get("reaction_source_role") in roles
        and result.get("reaction_content") in TARGETED
    )


def clip_predictions(
    rows: list[dict[str, Any]],
    *,
    include_authority: bool,
    allow_unresolved_visual_order: bool = False,
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for row in rows:
        item_id = str(row["item_id"])
        positive = (
            row.get("error") is None
            and candidate_positive(
                row.get("result") or {},
                include_authority=include_authority,
                allow_unresolved_visual_order=allow_unresolved_visual_order,
            )
        )
        result[item_id] = result.get(item_id, False) or positive
    return result


def evaluate(
    manual_rows: list[dict[str, Any]],
    model_rows: dict[str, list[dict[str, Any]]],
    *,
    include_authority: bool,
) -> dict[str, Any]:
    manual_by = {row["item_id"]: row for row in manual_rows}
    item_ids = sorted(manual_by)
    gold = np.asarray(
        [
            manual_target(
                manual_by[item_id], include_authority=include_authority
            )
            for item_id in item_ids
        ],
        dtype=int,
    )
    predictions = {
        name: clip_predictions(rows, include_authority=include_authority)
        for name, rows in model_rows.items()
    }
    for name, rows in model_rows.items():
        if any(
            "triggering_action_visible" in (row.get("result") or {})
            for row in rows
        ):
            predictions[f"{name}__diagnostic_visual_no_order"] = clip_predictions(
                rows,
                include_authority=include_authority,
                allow_unresolved_visual_order=True,
            )
    base_score_by: dict[str, np.ndarray] = {
        name: np.asarray(
            [prediction.get(item_id, False) for item_id in item_ids],
            dtype=float,
        )
        for name, prediction in predictions.items()
        if "__diagnostic_" not in name
    }
    score_by = {
        name: np.asarray(
            [prediction.get(item_id, False) for item_id in item_ids],
            dtype=float,
        )
        for name, prediction in predictions.items()
    }
    if len(base_score_by) >= 2:
        matrix = np.stack(list(base_score_by.values()))
        score_by["model_union"] = np.any(matrix > 0, axis=0).astype(float)
        score_by["model_intersection"] = np.all(matrix > 0, axis=0).astype(float)
    metrics = {
        name: metric_record(gold, score) for name, score in score_by.items()
    }
    errors: dict[str, list[dict[str, Any]]] = {}
    for name, score in score_by.items():
        errors[name] = [
            {
                "item_id": item_id,
                "gold": bool(truth),
                "predicted": bool(prediction),
                "manual_role": manual_by[item_id].get("reaction_source_role"),
                "manual_content": manual_by[item_id].get("reaction_content"),
                "manual_description": manual_by[item_id].get("description"),
            }
            for item_id, truth, prediction in zip(
                item_ids, gold, score >= 0.5, strict=True
            )
            if bool(truth) != bool(prediction)
        ]
    return {
        "items": len(item_ids),
        "positives": int(gold.sum()),
        "metrics": metrics,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument(
        "--model-output",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model_rows = {
        name: load_jsonl(Path(path)) for name, path in args.model_output
    }
    manual = load_jsonl(args.manual)
    report = {
        "kind": "witnessed_reaction_candidate_vlm_holdout_v1",
        "evaluation": "clip_level_any_positive_candidate",
        "strict_bystander": evaluate(
            manual, model_rows, include_authority=False
        ),
        "extended_third_party": evaluate(
            manual, model_rows, include_authority=True
        ),
        "policy": "audit_only_no_corpus_mutation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
