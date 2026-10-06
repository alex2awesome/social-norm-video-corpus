#!/usr/bin/env python3
"""Validate production retrieval queries against the frozen audiovisual audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from typing import Any

import yaml


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def evaluate(
    config: dict[str, Any],
    selection: dict[str, Any],
    manual_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if config.get("policy", {}).get("retrieval_only") is not True:
        raise ValueError("audited search queries must be retrieval-only")
    if config.get("policy", {}).get("query_text_is_never_a_label") is not True:
        raise ValueError("query text must never be a label")
    items = selection.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("selection is empty")
    manual = {str(row.get("uid") or ""): row for row in manual_rows}
    if not all(manual) or len(manual) != len(manual_rows):
        raise ValueError("manual audit has empty or duplicate uid")
    selected_uids = [str(row.get("uid") or "") for row in items]
    if not all(selected_uids) or len(selected_uids) != len(set(selected_uids)):
        raise ValueError("selection is not source-disjoint")
    if set(selected_uids) != set(manual):
        raise ValueError("manual audit does not exactly cover selection")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        pillar = str(item.get("pillar") or "")
        query = str(item.get("query") or "")
        if not pillar or not query:
            raise ValueError("selection row lacks pillar or query")
        judgment = manual[str(item["uid"])]
        if judgment.get("intended_pillar") != pillar:
            raise ValueError(f"pillar mismatch for {item['uid']}")
        if judgment.get("strict_target_pass") is True and (
            judgment.get("visual_scene") in {None, "", "no"}
            or judgment.get("label_support") in {None, "", "none", "text_only"}
        ):
            raise ValueError(f"strict pass lacks visual label support: {item['uid']}")
        grouped[(pillar, query)].append(judgment)

    query_metrics = {}
    for (pillar, query), rows in sorted(grouped.items()):
        passed = sum(row.get("strict_target_pass") is True for row in rows)
        query_metrics[f"{pillar}\t{query}"] = {
            "pillar": pillar,
            "query": query,
            "reviewed": len(rows),
            "strict_target_passes": passed,
            "strict_target_precision": passed / len(rows),
            "precision_wilson_95": wilson(passed, len(rows)),
        }

    configured = config.get("queries")
    if not isinstance(configured, list) or not configured:
        raise ValueError("no production queries configured")
    configured_keys = set()
    for row in configured:
        key = f"{row.get('pillar')}\t{row.get('query')}"
        if key in configured_keys:
            raise ValueError(f"duplicate configured query: {key}")
        configured_keys.add(key)
        if key not in query_metrics:
            raise ValueError(f"configured query was not audiovisually reviewed: {key}")
        actual = query_metrics[key]
        if row.get("reviewed") != actual["reviewed"]:
            raise ValueError(f"reviewed count mismatch: {key}")
        if row.get("strict_target_passes") != actual["strict_target_passes"]:
            raise ValueError(f"pass count mismatch: {key}")
        if actual["strict_target_passes"] <= 0:
            raise ValueError(f"zero-yield query cannot be active: {key}")
        if actual["pillar"] == "witnessed":
            raise ValueError("witnessed v4 search failed its pillar contract")

    selected_metrics = [query_metrics[key] for key in sorted(configured_keys)]
    reviewed = sum(row["reviewed"] for row in selected_metrics)
    passed = sum(row["strict_target_passes"] for row in selected_metrics)
    audit = config.get("audit", {})
    if audit.get("reviewed_sources") != len(items):
        raise ValueError("audit reviewed_sources mismatch")
    if audit.get("selected_query_reviewed_sources") != reviewed:
        raise ValueError("selected query reviewed denominator mismatch")
    if audit.get("selected_query_strict_target_passes") != passed:
        raise ValueError("selected query pass numerator mismatch")

    omitted = [value for key, value in query_metrics.items() if key not in configured_keys]
    unsafe_omissions = [
        row for row in omitted
        if row["strict_target_passes"] > 0 and row["pillar"] != "witnessed"
    ]
    if unsafe_omissions:
        raise ValueError("non-witnessed positive-yield audited queries were omitted")
    witnessed = [row for row in omitted if row["pillar"] == "witnessed"]
    zero_yield = [
        row for row in omitted
        if row["pillar"] != "witnessed" and row["strict_target_passes"] == 0
    ]
    if audit.get("excluded_zero_pass_query_formulations") != len(zero_yield):
        raise ValueError("excluded zero-pass query count mismatch")
    if audit.get("excluded_witnessed_query_formulations") != len(witnessed):
        raise ValueError("excluded witnessed query count mismatch")

    by_pillar = {}
    for pillar in sorted({row["pillar"] for row in selected_metrics}):
        rows = [row for row in selected_metrics if row["pillar"] == pillar]
        n = sum(row["reviewed"] for row in rows)
        k = sum(row["strict_target_passes"] for row in rows)
        by_pillar[pillar] = {
            "queries": len(rows),
            "reviewed": n,
            "strict_target_passes": k,
            "conditional_precision": k / n,
            "precision_wilson_95": wilson(k, n),
        }
    return {
        "kind": "audited_search_queries_v1_validation",
        "policy": "retrieval_only_never_label_or_acceptance",
        "source_disjoint_reviewed": len(items),
        "configured_queries": len(configured_keys),
        "configured_reviewed": reviewed,
        "configured_strict_target_passes": passed,
        "configured_conditional_precision": passed / reviewed,
        "configured_precision_wilson_95": wilson(passed, reviewed),
        "by_pillar": by_pillar,
        "omitted_zero_yield_queries": len(zero_yield),
        "omitted_witnessed_queries": len(witnessed),
        "query_metrics": query_metrics,
        "automatic_acceptance": False,
        "corpus_mutated": False,
    }


def run(config_path: Path, selection_path: Path, manual_path: Path, output: Path) -> dict[str, Any]:
    report = evaluate(
        yaml.safe_load(config_path.read_text()),
        json.loads(selection_path.read_text()),
        read_jsonl(manual_path),
    )
    report["artifact_sha256"] = {
        "config": sha256(config_path),
        "selection": sha256(selection_path),
        "manual": sha256(manual_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.selection, args.manual, args.out), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
