#!/usr/bin/env python3
"""Validate and summarize the preregistered conditioned-v5 manual audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
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


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = sum(bool(row["manual_positive"]) for row in rows)
    return {
        "n": len(rows),
        "manual_positive": positives,
        "manual_negative": len(rows) - positives,
        "rate": positives / len(rows) if rows else None,
        "wilson_95": wilson_95(positives, len(rows)),
    }


def grouped_metrics(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, ""))].append(row)
    return {
        key: metric(group_rows)
        for key, group_rows in sorted(groups.items())
    }


CONTEXT_ONLY_MECHANISMS = {
    "context_broll",
    "documentary_montage",
    "generic_broll",
    "generic_broll_and_explainer",
    "generic_montage",
    "news_montage",
    "news_or_context_broll",
    "presenter_plus_generic_broll",
    "presenter_plus_stock_graphics",
    "single_speaker_presentation",
    "talking_heads",
}
FORMAL_OR_PROCEDURAL_MECHANISMS = {
    "formal_civic_ritual",
    "formal_procedure",
    "formal_procedure_or_aftermath",
    "formal_safety_rule",
    "procedural_cleanliness",
    "procedural_financial_lesson",
}


def mechanism_family(mechanism: str) -> str:
    if mechanism in CONTEXT_ONLY_MECHANISMS:
        return "context_only_broll_montage_or_presentation"
    if mechanism in FORMAL_OR_PROCEDURAL_MECHANISMS:
        return "formal_or_procedural_not_informal_social_conduct"
    if mechanism == "ambiguous_story_excerpt":
        return "ambiguous_or_unlocalized_story_excerpt"
    raise ValueError(f"unmapped false-positive mechanism: {mechanism!r}")


def evaluate(
    selection: Path,
    decisions: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = read_jsonl(selection)
    judged = read_jsonl(decisions)
    if len(selected) != 90:
        raise ValueError(f"selection must contain 90 items, found {len(selected)}")
    if len(judged) != 90:
        raise ValueError(f"decisions must contain 90 items, found {len(judged)}")

    selected_by_index: dict[int, dict[str, Any]] = {}
    for row in selected:
        index = int(row["audit_index"])
        if index in selected_by_index:
            raise ValueError(f"duplicate selection audit_index {index}")
        selected_by_index[index] = row
    judged_by_index: dict[int, dict[str, Any]] = {}
    for row in judged:
        index = int(row["audit_index"])
        if index in judged_by_index:
            raise ValueError(f"duplicate decision audit_index {index}")
        if not isinstance(row.get("manual_positive"), bool):
            raise ValueError(f"audit_index {index}: manual_positive must be boolean")
        judged_by_index[index] = row
    expected = set(range(90))
    if set(selected_by_index) != expected:
        raise ValueError("selection audit indices must be exactly 0..89")
    if set(judged_by_index) != expected:
        raise ValueError("decision audit indices must be exactly 0..89")

    joined: list[dict[str, Any]] = []
    for index in range(90):
        selected_row = selected_by_index[index]
        decision = judged_by_index[index]
        band = selected_row.get("audit_band")
        if band not in {"dual_strict", "qwen_strict_glm_reject"}:
            raise ValueError(f"audit_index {index}: unexpected audit_band {band!r}")
        joined.append(
            {
                "audit_index": index,
                "audit_band": band,
                "item_id": selected_row["item_id"],
                "uid": selected_row["uid"],
                "category": selected_row.get("category"),
                "polarity": selected_row.get("polarity"),
                "norm": selected_row.get("norm"),
                **{key: value for key, value in decision.items() if key != "audit_index"},
            }
        )

    primary = [row for row in joined if row["audit_band"] == "dual_strict"]
    disagreement = [
        row for row in joined if row["audit_band"] == "qwen_strict_glm_reject"
    ]
    if len(primary) != 60 or len(disagreement) != 30:
        raise ValueError(
            "expected 60 dual_strict and 30 qwen_strict_glm_reject items"
        )
    false_positives = [row for row in primary if not row["manual_positive"]]
    mechanism_counts = Counter(row["mechanism"] for row in false_positives)
    mechanism_family_counts = Counter(
        mechanism_family(row["mechanism"]) for row in false_positives
    )
    largest_mechanism_count = max(mechanism_family_counts.values(), default=0)
    primary_metric = metric(primary)
    pass_checks = {
        "at_least_30_conclusive": len(primary) >= 30,
        "all_selected_manually_judged": len(joined) == 90,
        "raw_precision_at_least_90pct": primary_metric["rate"] >= 0.90,
        "no_false_positive_mechanism_over_5pct_of_primary": (
            largest_mechanism_count / len(primary) <= 0.05
        ),
    }
    summary = {
        "kind": "instructional_conditioned_v5_corpus_validation",
        "selection": str(selection),
        "selection_sha256": sha256(selection),
        "decisions": str(decisions),
        "decisions_sha256": sha256(decisions),
        "bands": {
            "dual_strict_primary_precision": primary_metric,
            "qwen_strict_glm_reject_recovery": metric(disagreement),
        },
        "primary_by_category": grouped_metrics(primary, "category"),
        "primary_by_polarity": grouped_metrics(primary, "polarity"),
        "primary_false_positive_mechanisms": dict(
            sorted(mechanism_counts.items())
        ),
        "primary_false_positive_mechanism_families": dict(
            sorted(mechanism_family_counts.items())
        ),
        "largest_primary_false_positive_mechanism_family": {
            "count": largest_mechanism_count,
            "fraction_of_primary": largest_mechanism_count / len(primary),
        },
        "promotion_checks": pass_checks,
        "promotion_pass": all(pass_checks.values()),
        "note": (
            "The disagreement cohort was deliberately sampled and its recovery "
            "rate is not a corpus prevalence estimate."
        ),
    }
    return joined, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--out-ledger", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    args = parser.parse_args()
    joined, summary = evaluate(args.selection, args.decisions)
    args.out_ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in joined)
    )
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["bands"], sort_keys=True))
    print(json.dumps({"promotion_pass": summary["promotion_pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
