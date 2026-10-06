#!/usr/bin/env python3
"""Evaluate the preregistered prospective instructional V10 audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


YES_NO = {"Y", "N"}
VISUAL = {"D", "N"}
POLARITIES = {"violation", "correct", "neutral", "mixed", "none"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_95(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return [center - radius, center + radius]


def metric(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    positive = sum(bool(predicate(row)) for row in rows)
    return {
        "n": len(rows),
        "positive": positive,
        "negative": len(rows) - positive,
        "rate": positive / len(rows) if rows else None,
        "wilson_95": wilson_95(positive, len(rows)),
    }


def indexed(
    rows: list[dict[str, Any]],
    *,
    name: str,
    expected: set[int],
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            index = int(row["audit_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{name}: invalid audit_index") from exc
        if index in result:
            raise ValueError(f"{name}: duplicate audit_index {index}")
        result[index] = row
    if set(result) != expected:
        raise ValueError(
            f"{name}: coverage mismatch; "
            f"missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    return result


def evaluate(
    selection_path: Path,
    visual_path: Path,
    semantic_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection_rows = read_jsonl(selection_path)
    selection_indices = {int(row["audit_index"]) for row in selection_rows}
    if len(selection_indices) != len(selection_rows):
        raise ValueError("selection: duplicate audit_index")
    expected = set(range(len(selection_rows)))
    if selection_indices != expected:
        raise ValueError("selection audit indices must be exactly 0..N-1")
    selection = indexed(selection_rows, name="selection", expected=expected)
    visual = indexed(read_tsv(visual_path), name="visual", expected=expected)
    semantic = indexed(read_tsv(semantic_path), name="semantic", expected=expected)

    joined: list[dict[str, Any]] = []
    for index in range(len(selection_rows)):
        source = selection[index]
        visual_row = visual[index]
        semantic_row = semantic[index]
        visual_demo = visual_row.get("visual_demo")
        exact = semantic_row.get("exact_original")
        relabel = semantic_row.get("usable_after_relabel")
        polarity = semantic_row.get("visible_polarity")
        failure = semantic_row.get("failure_mechanism", "")
        if source.get("band") not in {"primary_candidate", "control_reject"}:
            raise ValueError(f"audit_index {index}: invalid band")
        if visual_demo not in VISUAL:
            raise ValueError(f"audit_index {index}: invalid visual_demo")
        if exact not in YES_NO or relabel not in YES_NO:
            raise ValueError(f"audit_index {index}: invalid semantic Y/N")
        if polarity not in POLARITIES:
            raise ValueError(f"audit_index {index}: invalid visible_polarity")
        if exact == "Y" and (relabel != "Y" or failure != "none"):
            raise ValueError(
                f"audit_index {index}: exact item must be relabel-usable "
                "with failure=none"
            )
        if exact == "N" and failure in {"", "none"}:
            raise ValueError(
                f"audit_index {index}: inexact item needs a failure mechanism"
            )
        if visual_demo == "N" and (exact != "N" or relabel != "N"):
            raise ValueError(
                f"audit_index {index}: no-demo item cannot be label-usable"
            )
        joined.append(
            {
                "audit_index": index,
                "item_id": source["item_id"],
                "uid": source["uid"],
                "band": source["band"],
                "candidate_stage": source["candidate_stage"],
                "category": source.get("category"),
                "norm": source.get("norm"),
                "polarity": source.get("polarity"),
                "manual_visual_demo": visual_demo == "D",
                "manual_exact_original": exact == "Y",
                "manual_usable_after_relabel": relabel == "Y",
                "manual_visible_polarity": polarity,
                "manual_failure_mechanism": failure,
                "manual_semantic_note": semantic_row.get("semantic_notes"),
                "manual_literal_description": visual_row.get(
                    "literal_description"
                ),
                "manual_review_basis": visual_row.get("review_basis"),
                "manual_blind_status": visual_row.get("blind_status"),
            }
        )

    candidates = [
        row for row in joined if row["band"] == "primary_candidate"
    ]
    controls = [row for row in joined if row["band"] == "control_reject"]
    if any(row["candidate_stage"] != "candidate" for row in candidates):
        raise ValueError("primary rows must have candidate_stage=candidate")

    pred_visual = lambda row: bool(row["manual_visual_demo"])
    pred_exact = lambda row: bool(row["manual_exact_original"])
    pred_relabel = lambda row: bool(row["manual_usable_after_relabel"])
    failure_counts = Counter(
        row["manual_failure_mechanism"]
        for row in candidates
        if not row["manual_exact_original"]
    )
    max_failure_count = max(failure_counts.values(), default=0)
    distinct_candidate_uids = len({row["uid"] for row in candidates})

    controls_by_stage: dict[str, dict[str, Any]] = {}
    stage_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in controls:
        stage_rows[row["candidate_stage"]].append(row)
    for stage, rows in sorted(stage_rows.items()):
        controls_by_stage[stage] = {
            "visual_demo": metric(rows, pred_visual),
            "exact_original": metric(rows, pred_exact),
            "usable_after_relabel": metric(rows, pred_relabel),
        }

    candidate_visual = metric(candidates, pred_visual)
    candidate_exact = metric(candidates, pred_exact)
    candidate_relabel = metric(candidates, pred_relabel)
    checks = {
        "at_least_30_candidate_clips": len(candidates) >= 30,
        "at_least_20_distinct_candidate_uids": distinct_candidate_uids >= 20,
        "all_outputs_conclusive": all(
            row["manual_visible_polarity"] in POLARITIES for row in candidates
        ),
        "candidate_visual_demo_at_least_95_percent": (
            candidate_visual["rate"] is not None
            and candidate_visual["rate"] >= 0.95
        ),
        "candidate_exact_at_least_90_percent": (
            candidate_exact["rate"] is not None
            and candidate_exact["rate"] >= 0.90
        ),
        "candidate_source_exact_at_least_90_percent": (
            candidate_exact["rate"] is not None
            and candidate_exact["rate"] >= 0.90
        ),
        "no_single_failure_mechanism_over_5_percent": (
            not candidates or max_failure_count / len(candidates) <= 0.05
        ),
    }
    passed = all(checks.values())
    summary = {
        "kind": "instructional_v10_prospective_youtube_validation",
        "coverage": {
            "audited_total": len(joined),
            "candidate_clips": len(candidates),
            "candidate_distinct_uids": distinct_candidate_uids,
            "control_rejects": len(controls),
            "control_stage_counts": dict(
                sorted(Counter(row["candidate_stage"] for row in controls).items())
            ),
        },
        "candidate_metrics": {
            "visual_demo": candidate_visual,
            "exact_original": candidate_exact,
            "usable_after_relabel": candidate_relabel,
        },
        "candidate_exact_failure_mechanisms": dict(sorted(failure_counts.items())),
        "candidate_false_positive_indices": [
            row["audit_index"]
            for row in candidates
            if not row["manual_exact_original"]
        ],
        "control_metrics": {
            "visual_demo": metric(controls, pred_visual),
            "exact_original": metric(controls, pred_exact),
            "usable_after_relabel": metric(controls, pred_relabel),
            "by_rejection_stage": controls_by_stage,
        },
        "preregistered_checks": checks,
        "preregistered_pass": passed,
        "promotion_eligible": False,
        "decision": (
            "FAIL: the frozen V10 conjunction did not satisfy the "
            "preregistered prospective requirements."
            if not passed
            else (
                "PASS AS SHADOW RANKING ONLY: all preregistered metrics passed, "
                "but no automatic corpus mutation is authorized."
            )
        ),
        "artifact_sha256": {
            "selection": sha256(selection_path),
            "manual_visual_ledger": sha256(visual_path),
            "manual_semantic_ledger": sha256(semantic_path),
        },
    }
    return joined, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--visual-ledger", type=Path, required=True)
    parser.add_argument("--semantic-ledger", type=Path, required=True)
    parser.add_argument("--evaluated-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    joined, summary = evaluate(
        args.selection, args.visual_ledger, args.semantic_ledger
    )
    args.evaluated_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in joined)
    )
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
