#!/usr/bin/env python3
"""Evaluate the frozen dense manual audit of the instructional V8 rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


EXPECTED_BANDS = {
    "v8_dual_strict": 60,
    "v8_conjunction_reject": 30,
}
SEMANTIC_STATUSES = {"exact", "usable_after_relabel", "unusable"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_95(successes: int, total: int) -> list[float] | None:
    if total == 0:
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


def metric(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    positives = sum(predicate(row) for row in rows)
    return {
        "n": len(rows),
        "positive": positives,
        "negative": len(rows) - positives,
        "rate": positives / len(rows) if rows else None,
        "wilson_95": wilson_95(positives, len(rows)),
    }


def grouped_metrics(
    rows: list[dict[str, Any]],
    field: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, ""))].append(row)
    return {
        key: metric(group_rows, predicate)
        for key, group_rows in sorted(groups.items())
    }


def _indexed(rows: list[dict[str, Any]], source: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["audit_index"])
        if index in result:
            raise ValueError(f"{source}: duplicate audit_index {index}")
        result[index] = row
    expected = set(range(90))
    if set(result) != expected:
        raise ValueError(f"{source}: audit indices must be exactly 0..89")
    return result


def _semantic_statuses(
    visual_by_index: dict[int, dict[str, Any]],
    semantic: dict[str, Any],
) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
    if not semantic.get("all_90_rows_reviewed_after_visual_freeze"):
        raise ValueError("semantic audit is not marked complete")
    defaults = {
        "situated_social_event": "exact",
        "recoverable_explicit_etiquette_demo": "exact",
        "no_visual_demo": "unusable",
    }
    overrides: dict[int, dict[str, Any]] = {}
    for override in semantic.get("overrides", []):
        index = int(override["audit_index"])
        if index in overrides:
            raise ValueError(f"duplicate semantic override {index}")
        status = override.get("status")
        if status not in SEMANTIC_STATUSES:
            raise ValueError(f"audit_index {index}: invalid semantic status {status!r}")
        overrides[index] = override

    statuses: dict[int, str] = {}
    for index, row in visual_by_index.items():
        band = row.get("visual_band")
        if band not in defaults:
            raise ValueError(f"audit_index {index}: invalid visual_band {band!r}")
        statuses[index] = overrides.get(index, {}).get("status", defaults[band])
        if band == "no_visual_demo" and statuses[index] != "unusable":
            raise ValueError(f"audit_index {index}: no-visual item cannot be usable")
    return statuses, overrides


def evaluate(
    selection_path: Path,
    visual_path: Path,
    semantic_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = _indexed(read_jsonl(selection_path), "selection")
    visual = _indexed(read_jsonl(visual_path), "visual")
    semantic = json.loads(semantic_path.read_text())
    statuses, overrides = _semantic_statuses(visual, semantic)

    joined: list[dict[str, Any]] = []
    for index in range(90):
        selected_row = selected[index]
        visual_row = visual[index]
        expected_candidate = f"candidate:{index:02d}"
        if visual_row.get("item_id") != expected_candidate:
            raise ValueError(f"audit_index {index}: unexpected candidate ID")
        band = selected_row.get("audit_band")
        if band not in EXPECTED_BANDS:
            raise ValueError(f"audit_index {index}: invalid audit band {band!r}")
        strict_visual = visual_row.get("manual_usable") == "yes"
        any_visual_demo = visual_row.get("visual_band") != "no_visual_demo"
        status = statuses[index]
        joined.append(
            {
                "audit_index": index,
                "candidate_id": expected_candidate,
                "item_id": selected_row["item_id"],
                "uid": selected_row["uid"],
                "audit_band": band,
                "category": selected_row.get("category"),
                "polarity": selected_row.get("polarity"),
                "norm": selected_row.get("norm"),
                "title": selected_row.get("title"),
                "strict_situated_event": strict_visual,
                "any_visual_demo": any_visual_demo,
                "semantic_status": status,
                "exact_weak_label_usable": status == "exact",
                "usable_after_relabel": status in {"exact", "usable_after_relabel"},
                "visual_band": visual_row["visual_band"],
                "visual_failure_mechanism": visual_row.get("failure_mechanism"),
                "literal_description": visual_row["literal_description"],
                "semantic_override": overrides.get(index),
            }
        )

    for band, expected_count in EXPECTED_BANDS.items():
        actual = sum(row["audit_band"] == band for row in joined)
        if actual != expected_count:
            raise ValueError(f"{band}: expected {expected_count}, found {actual}")

    primary = [row for row in joined if row["audit_band"] == "v8_dual_strict"]
    rejected = [
        row for row in joined
        if row["audit_band"] == "v8_conjunction_reject"
    ]
    strict = lambda row: bool(row["strict_situated_event"])
    any_demo = lambda row: bool(row["any_visual_demo"])
    exact = lambda row: bool(row["exact_weak_label_usable"])
    relabel = lambda row: bool(row["usable_after_relabel"])

    visual_false_positives = [row for row in primary if not strict(row)]
    visual_failure_counts = Counter(
        row["visual_failure_mechanism"] for row in visual_false_positives
    )
    largest_visual_mechanism = max(visual_failure_counts.values(), default=0)
    primary_strict_metric = metric(primary, strict)
    promotion_checks = {
        "all_90_manually_reviewed": len(joined) == 90,
        "primary_has_60_items": len(primary) == 60,
        "strict_visual_precision_at_least_90pct": (
            primary_strict_metric["rate"] >= 0.90
        ),
        "no_visual_failure_mechanism_over_5pct_of_primary": (
            largest_visual_mechanism / len(primary) <= 0.05
        ),
    }

    summary = {
        "kind": "instructional_v8_fresh_dense36_validation",
        "inputs": {
            "selection": str(selection_path),
            "selection_sha256": sha256(selection_path),
            "visual_adjudication": str(visual_path),
            "visual_adjudication_sha256": sha256(visual_path),
            "semantic_adjudication": str(semantic_path),
            "semantic_adjudication_sha256": sha256(semantic_path),
        },
        "bands": {
            "v8_dual_strict": {
                "strict_situated_event": primary_strict_metric,
                "any_visual_demo": metric(primary, any_demo),
                "exact_weak_label_usable": metric(primary, exact),
                "usable_after_relabel": metric(primary, relabel),
            },
            "v8_conjunction_reject": {
                "strict_situated_event": metric(rejected, strict),
                "any_visual_demo": metric(rejected, any_demo),
                "exact_weak_label_usable": metric(rejected, exact),
                "usable_after_relabel": metric(rejected, relabel),
            },
            "all_90": {
                "strict_situated_event": metric(joined, strict),
                "any_visual_demo": metric(joined, any_demo),
                "exact_weak_label_usable": metric(joined, exact),
                "usable_after_relabel": metric(joined, relabel),
            },
        },
        "primary_by_category": {
            "strict_situated_event": grouped_metrics(primary, "category", strict),
            "usable_after_relabel": grouped_metrics(primary, "category", relabel),
        },
        "primary_by_polarity": {
            "strict_situated_event": grouped_metrics(primary, "polarity", strict),
            "usable_after_relabel": grouped_metrics(primary, "polarity", relabel),
        },
        "primary_visual_false_positive_mechanisms": dict(
            sorted(visual_failure_counts.items())
        ),
        "semantic_status_counts": dict(
            sorted(Counter(row["semantic_status"] for row in joined).items())
        ),
        "promotion_checks": promotion_checks,
        "promotion_pass": all(promotion_checks.values()),
        "decision": (
            "FAIL: keep V8 shadow-only; the dual conjunction did not reach the "
            "preregistered 90% strict visual precision and rejected many usable demos."
        ),
        "notes": [
            "The 30-item reject cohort was deliberately sampled and is not a corpus prevalence estimate.",
            "The explicit-etiquette salvage band honors the requirement that any format is acceptable when it contains a clear demonstration.",
            "No source clip or corpus membership was changed by this evaluation.",
        ],
    }
    return joined, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--visual", required=True, type=Path)
    parser.add_argument("--semantic", required=True, type=Path)
    parser.add_argument("--out-ledger", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    args = parser.parse_args()
    joined, summary = evaluate(args.selection, args.visual, args.semantic)
    args.out_ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in joined)
    )
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["bands"], sort_keys=True))
    print(json.dumps({"promotion_pass": summary["promotion_pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
