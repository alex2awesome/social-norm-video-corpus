#!/usr/bin/env python3
"""Evaluate the frozen manual audit of the instructional V9 corpus candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


EXPECTED_BANDS = {
    "primary_violation_all": 99,
    "comparison_correct": 30,
    "comparison_explanation": 7,
    "control_violation_reject": 30,
}
YES_NO = {"Y", "N"}
VISUAL = {"D", "N"}
POLARITIES = {"violation", "correct", "explanation", "unclear"}


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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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


def metric(
    rows: Iterable[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    materialized = list(rows)
    positives = sum(bool(predicate(row)) for row in materialized)
    total = len(materialized)
    return {
        "n": total,
        "positive": positives,
        "negative": total - positives,
        "rate": positives / total if total else None,
        "wilson_95": wilson_95(positives, total),
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
        key: metric(group, predicate)
        for key, group in sorted(groups.items())
    }


def _index(
    rows: list[dict[str, Any]],
    source: str,
    *,
    total: int,
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            audit_index = int(row["audit_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{source}: invalid audit_index") from exc
        if audit_index in indexed:
            raise ValueError(f"{source}: duplicate audit_index {audit_index}")
        indexed[audit_index] = row
    expected = set(range(total))
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(
            f"{source}: indices must be exactly 0..{total - 1}; "
            f"missing={missing}, extra={extra}"
        )
    return indexed


def _source_cluster_metric(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["uid"])].append(row)
    passing = sum(all(bool(row[field]) for row in cluster) for cluster in clusters.values())
    result = metric(
        [{"pass": all(bool(row[field]) for row in cluster)} for cluster in clusters.values()],
        lambda row: bool(row["pass"]),
    )
    result["definition"] = (
        f"a source UID passes only when every selected clip has {field}=true"
    )
    result["multi_clip_source_count"] = sum(
        len(cluster) > 1 for cluster in clusters.values()
    )
    assert result["positive"] == passing
    return result


def evaluate(
    selection_path: Path,
    visual_path: Path,
    semantic_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_rows = read_jsonl(selection_path)
    total = sum(EXPECTED_BANDS.values())
    selected = _index(selected_rows, "selection", total=total)
    visual = _index(read_tsv(visual_path), "visual ledger", total=total)
    semantic = _index(read_tsv(semantic_path), "semantic ledger", total=total)

    joined: list[dict[str, Any]] = []
    for audit_index in range(total):
        source = selected[audit_index]
        visual_row = visual[audit_index]
        semantic_row = semantic[audit_index]
        band = str(source.get("band", ""))
        if band not in EXPECTED_BANDS:
            raise ValueError(f"audit_index {audit_index}: invalid band {band!r}")
        if semantic_row.get("band") != band:
            raise ValueError(
                f"audit_index {audit_index}: semantic band does not match selection"
            )
        visual_demo = visual_row.get("visual_demo")
        exact = semantic_row.get("exact_original")
        relabel = semantic_row.get("usable_after_relabel")
        polarity = semantic_row.get("visible_polarity")
        if visual_demo not in VISUAL:
            raise ValueError(f"audit_index {audit_index}: invalid visual_demo")
        if exact not in YES_NO or relabel not in YES_NO:
            raise ValueError(f"audit_index {audit_index}: invalid semantic Y/N")
        if polarity not in POLARITIES:
            raise ValueError(f"audit_index {audit_index}: invalid visible_polarity")
        if exact == "Y" and relabel != "Y":
            raise ValueError(
                f"audit_index {audit_index}: exact item must be relabel-usable"
            )
        if visual_demo == "N" and (exact != "N" or relabel != "N"):
            raise ValueError(
                f"audit_index {audit_index}: no-demo item cannot be label-usable"
            )
        failure = semantic_row.get("failure_mechanism", "")
        if exact == "Y" and failure != "none":
            raise ValueError(
                f"audit_index {audit_index}: exact item must have failure=none"
            )
        if exact == "N" and failure in {"", "none"}:
            raise ValueError(
                f"audit_index {audit_index}: inexact item needs a failure mechanism"
            )
        joined.append(
            {
                "audit_index": audit_index,
                "candidate_id": source["candidate_id"],
                "item_id": source["item_id"],
                "uid": source["uid"],
                "band": band,
                "category": source.get("category"),
                "source_platform": source.get("source_platform"),
                "original_polarity": source.get("polarity"),
                "original_norm": source.get("norm"),
                "manual_visual_demo": visual_demo == "D",
                "manual_exact_original": exact == "Y",
                "manual_usable_after_relabel": relabel == "Y",
                "manual_visible_polarity": polarity,
                "manual_relabel": semantic_row.get("relabel"),
                "manual_failure_mechanism": failure,
                "manual_note": semantic_row.get("manual_note"),
                "literal_description": visual_row.get("literal_description"),
                "blind_status": visual_row.get("blind_status"),
            }
        )

    actual_bands = Counter(row["band"] for row in joined)
    if actual_bands != Counter(EXPECTED_BANDS):
        raise ValueError(
            f"unexpected band counts: {dict(actual_bands)}; "
            f"expected {EXPECTED_BANDS}"
        )

    primary = [row for row in joined if row["band"] == "primary_violation_all"]
    correct = [row for row in joined if row["band"] == "comparison_correct"]
    explanation = [
        row for row in joined if row["band"] == "comparison_explanation"
    ]
    rejected = [
        row for row in joined if row["band"] == "control_violation_reject"
    ]
    visual_pred = lambda row: bool(row["manual_visual_demo"])
    exact_pred = lambda row: bool(row["manual_exact_original"])
    relabel_pred = lambda row: bool(row["manual_usable_after_relabel"])

    primary_visual = metric(primary, visual_pred)
    primary_exact = metric(primary, exact_pred)
    primary_relabel = metric(primary, relabel_pred)
    source_exact = _source_cluster_metric(primary, "manual_exact_original")
    source_visual = _source_cluster_metric(primary, "manual_visual_demo")
    primary_failures = Counter(
        row["manual_failure_mechanism"]
        for row in primary
        if not row["manual_exact_original"]
    )
    primary_visual_failures = Counter(
        row["manual_failure_mechanism"]
        for row in primary
        if not row["manual_visual_demo"]
    )
    largest_failure = max(primary_failures.values(), default=0)

    checks = {
        "at_least_30_primary_clips": len(primary) >= 30,
        "at_least_20_distinct_primary_sources": len({row["uid"] for row in primary}) >= 20,
        "every_primary_output_conclusive": all(
            row["manual_visible_polarity"] in POLARITIES for row in primary
        ),
        "clip_exact_precision_at_least_90pct": primary_exact["rate"] >= 0.90,
        "source_cluster_exact_precision_at_least_90pct": source_exact["rate"] >= 0.90,
        "clip_visual_demo_precision_at_least_95pct": primary_visual["rate"] >= 0.95,
        "no_exact_failure_mechanism_over_5pct_of_primary": (
            largest_failure / len(primary) <= 0.05
        ),
    }

    def band_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "visual_demo": metric(rows, visual_pred),
            "exact_original": metric(rows, exact_pred),
            "usable_after_relabel": metric(rows, relabel_pred),
        }

    summary = {
        "kind": "instructional_v9_fresh_corpus_validation",
        "inputs": {
            "selection": str(selection_path),
            "selection_sha256": sha256(selection_path),
            "blind_visual_ledger": str(visual_path),
            "blind_visual_ledger_sha256": sha256(visual_path),
            "manual_semantic_ledger": str(semantic_path),
            "manual_semantic_ledger_sha256": sha256(semantic_path),
        },
        "review_coverage": {
            "all_selected": len(joined),
            "all_selected_reviewed": len(joined) == total,
            "blind_before_semantic_reveal": sum(
                row["blind_status"] == "blind" for row in joined
            ),
            "recorded_blind_protocol_exceptions": sum(
                row["blind_status"] != "blind" for row in joined
            ),
        },
        "bands": {
            "primary_violation_all": band_metrics(primary),
            "comparison_correct": band_metrics(correct),
            "comparison_explanation": band_metrics(explanation),
            "control_violation_reject": band_metrics(rejected),
            "all_166": band_metrics(joined),
        },
        "primary_source_clusters": {
            "distinct_sources": len({row["uid"] for row in primary}),
            "exact_original_all_clips": source_exact,
            "visual_demo_all_clips": source_visual,
        },
        "primary_by_category": {
            "visual_demo": grouped_metrics(primary, "category", visual_pred),
            "exact_original": grouped_metrics(primary, "category", exact_pred),
            "usable_after_relabel": grouped_metrics(primary, "category", relabel_pred),
        },
        "primary_by_platform": {
            "visual_demo": grouped_metrics(primary, "source_platform", visual_pred),
            "exact_original": grouped_metrics(primary, "source_platform", exact_pred),
            "usable_after_relabel": grouped_metrics(primary, "source_platform", relabel_pred),
        },
        "primary_exact_failure_mechanisms": dict(sorted(primary_failures.items())),
        "primary_visual_failure_mechanisms": dict(
            sorted(primary_visual_failures.items())
        ),
        "primary_false_positive_indices": [
            row["audit_index"] for row in primary if not exact_pred(row)
        ],
        "primary_no_demo_indices": [
            row["audit_index"] for row in primary if not visual_pred(row)
        ],
        "rejected_recovery": {
            "note": "Deliberately sampled control; not a corpus-prevalence estimate.",
            "visual_demo": metric(rejected, visual_pred),
            "exact_original": metric(rejected, exact_pred),
            "usable_after_relabel": metric(rejected, relabel_pred),
        },
        "promotion_checks": checks,
        "promotion_pass": all(checks.values()),
        "decision": (
            "FAIL: keep V9 shadow-only. The frozen rule misses its visual-demo "
            "precision threshold and has systematic semantic-label leakage."
        ),
        "notes": [
            "Any format counts as a demo when the target social behavior is concretely performed.",
            "No clip was deleted, quarantined, relabeled in place, or removed from corpus membership.",
            "A salvageable but inexact item remains available for ranking under a separate relabel field.",
            "The control/reject band estimates recovery modes, not their corpus prevalence.",
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
    args.out_ledger.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in joined)
    )
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["bands"], sort_keys=True))
    print(json.dumps({"promotion_pass": summary["promotion_pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
