#!/usr/bin/env python3
"""Evaluate all instructional V10 discovery outputs against frozen manual gold."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_successful(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("error") is None and row.get("result") is not None:
            latest[row["item_id"]] = row
    return latest


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


def selection_metric(
    rows: list[dict[str, Any]],
    selected: Callable[[dict[str, Any]], bool],
    truth: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    chosen = [row for row in rows if selected(row)]
    true_positive = sum(bool(truth(row)) for row in chosen)
    all_positive = sum(bool(truth(row)) for row in rows)
    return {
        "selected": len(chosen),
        "true_positive": true_positive,
        "false_positive": len(chosen) - true_positive,
        "precision": true_positive / len(chosen) if chosen else None,
        "precision_wilson_95": wilson_95(true_positive, len(chosen)),
        "recall": true_positive / all_positive if all_positive else None,
        "false_positive_indices": [
            row["audit_index"] for row in chosen if not truth(row)
        ],
        "false_negative_indices": [
            row["audit_index"]
            for row in rows
            if truth(row) and not selected(row)
        ],
    }


def evaluate(
    selection_path: Path,
    visual_gold_path: Path,
    semantic_gold_path: Path,
    v10a_path: Path,
    v10b_path: Path,
    v10c_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = load_jsonl(selection_path)
    if len(selected) != 166:
        raise ValueError(f"expected 166 selection rows, found {len(selected)}")
    visual_gold = {
        int(row["audit_index"]): row for row in load_tsv(visual_gold_path)
    }
    semantic_gold = {
        int(row["audit_index"]): row for row in load_tsv(semantic_gold_path)
    }
    if set(visual_gold) != set(range(166)) or set(semantic_gold) != set(range(166)):
        raise ValueError("manual ledgers must cover exactly audit indices 0..165")
    v10a = latest_successful(v10a_path)
    v10b = latest_successful(v10b_path)
    v10c = latest_successful(v10c_path)
    item_ids = {row["item_id"] for row in selected}
    for name, records in [("V10A", v10a), ("V10B", v10b), ("V10C", v10c)]:
        if set(records) != item_ids:
            raise ValueError(
                f"{name} coverage mismatch: missing={len(item_ids - set(records))}, "
                f"extra={len(set(records) - item_ids)}"
            )

    joined: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for source in selected:
        index = int(source["audit_index"])
        if index in seen_indices:
            raise ValueError(f"duplicate audit_index {index}")
        seen_indices.add(index)
        item_id = source["item_id"]
        a = v10a[item_id]["result"]
        b = v10b[item_id]["result"]
        c = v10c[item_id]["result"]
        joined.append(
            {
                "audit_index": index,
                "item_id": item_id,
                "uid": source["uid"],
                "band": source["band"],
                "category": source.get("category"),
                "source_platform": source.get("source_platform"),
                "manual_visual_demo": visual_gold[index]["visual_demo"] == "D",
                "manual_exact_original": (
                    semantic_gold[index]["exact_original"] == "Y"
                ),
                "manual_usable_after_relabel": (
                    semantic_gold[index]["usable_after_relabel"] == "Y"
                ),
                "manual_failure_mechanism": semantic_gold[index][
                    "failure_mechanism"
                ],
                "v10a_demo": a["demo_usable"] == "yes",
                "v10a_scope": a["social_scope"],
                "v10a_scene_role": a["scene_role"],
                "v10b_exact": b["exact_original_usable"] == "yes",
                "v10b_relabel": b["usable_after_relabel"] == "yes",
                "v10b_specificity": b["proposed_norm_specificity"],
                "v10b_failure": b["failure_mechanism"],
                "v10c_strict": c["strict_exact_candidate"] == "yes",
                "v10c_relabel": c["usable_after_relabel"] == "yes",
                "v10c_hard_exclusion": c["hard_exclusion"],
                "v10c_norm_granularity": c["norm_granularity"],
                "v10c_failure": c["failure_mechanism"],
                "v10c_strict_youtube": (
                    c["strict_exact_candidate"] == "yes"
                    and source.get("source_platform") == "youtube"
                ),
            }
        )
    joined.sort(key=lambda row: row["audit_index"])
    if seen_indices != set(range(166)):
        raise ValueError("selection audit indices must be exactly 0..165")

    cohorts = {
        "all_166": joined,
        "primary_violation_all": [
            row for row in joined if row["band"] == "primary_violation_all"
        ],
        "comparison_correct": [
            row for row in joined if row["band"] == "comparison_correct"
        ],
        "comparison_explanation": [
            row for row in joined if row["band"] == "comparison_explanation"
        ],
        "control_violation_reject": [
            row for row in joined if row["band"] == "control_violation_reject"
        ],
    }
    rules = {
        "v10a_demo": lambda row: row["v10a_demo"],
        "v10b_exact": lambda row: row["v10b_exact"],
        "v10b_relabel": lambda row: row["v10b_relabel"],
        "v10c_strict": lambda row: row["v10c_strict"],
        "v10c_relabel": lambda row: row["v10c_relabel"],
        "v10c_strict_youtube": lambda row: row["v10c_strict_youtube"],
    }
    truth_fields = {
        "visual_demo": lambda row: row["manual_visual_demo"],
        "exact_original": lambda row: row["manual_exact_original"],
        "usable_after_relabel": lambda row: row[
            "manual_usable_after_relabel"
        ],
    }
    metrics: dict[str, Any] = {}
    for cohort_name, cohort in cohorts.items():
        metrics[cohort_name] = {
            rule_name: {
                truth_name: selection_metric(cohort, rule, truth)
                for truth_name, truth in truth_fields.items()
            }
            for rule_name, rule in rules.items()
        }

    primary = cohorts["primary_violation_all"]
    primary_exact_failure_counts = Counter(
        row["manual_failure_mechanism"]
        for row in primary
        if row["v10c_strict"] and not row["manual_exact_original"]
    )
    disagreement_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in joined:
        disagreement_matrix[
            "v10a_demo_vs_manual_visual"
        ][f"{row['v10a_demo']}/{row['manual_visual_demo']}"] += 1
        disagreement_matrix[
            "v10c_strict_vs_manual_exact"
        ][f"{row['v10c_strict']}/{row['manual_exact_original']}"] += 1

    summary = {
        "kind": "instructional_v10_revealed_discovery",
        "coverage": {
            "selection": len(selected),
            "v10a_successful_items": len(v10a),
            "v10b_successful_items": len(v10b),
            "v10c_successful_items": len(v10c),
            "manual_visual_items": len(visual_gold),
            "manual_semantic_items": len(semantic_gold),
        },
        "metrics": metrics,
        "primary_v10c_strict_false_positive_mechanisms": dict(
            sorted(primary_exact_failure_counts.items())
        ),
        "disagreement_matrix": {
            key: dict(sorted(values.items()))
            for key, values in disagreement_matrix.items()
        },
        "promotion_eligible": False,
        "decision": (
            "DISCOVERY ONLY: no V10 result can be promoted because every rule "
            "was developed after the 166-item manual labels were revealed."
        ),
        "next_required_step": (
            "Freeze one candidate and audit every output plus sampled rejects "
            "on newly collected, UID-disjoint clips."
        ),
    }
    return joined, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--visual-gold", type=Path, required=True)
    parser.add_argument("--semantic-gold", type=Path, required=True)
    parser.add_argument("--v10a", type=Path, required=True)
    parser.add_argument("--v10b", type=Path, required=True)
    parser.add_argument("--v10c", type=Path, required=True)
    parser.add_argument("--out-ledger", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()
    joined, summary = evaluate(
        args.selection,
        args.visual_gold,
        args.semantic_gold,
        args.v10a,
        args.v10b,
        args.v10c,
    )
    args.out_ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in joined)
    )
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    primary = summary["metrics"]["primary_violation_all"]
    print(
        json.dumps(
            {
                rule: {
                    "visual": values["visual_demo"]["precision"],
                    "exact": values["exact_original"]["precision"],
                    "exact_recall": values["exact_original"]["recall"],
                }
                for rule, values in primary.items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
