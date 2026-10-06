#!/usr/bin/env python3
"""Select a sealed, source-disjoint transfer cohort from prior commentary audits.

The selector stores hash-linked pointers to prior immutable manifests/results,
not duplicate transcript payloads.  It balances prior accepts and rejects so a
new blind manual pass can measure both over-acceptance and misses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ACCEPTED = {"accept", "accept_after_relabel"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(seed: str, *values: str) -> str:
    return hashlib.sha256((seed + "\0" + "\0".join(values)).encode()).hexdigest()


def load_excluded_uids(path: Path) -> set[str]:
    rows = read_jsonl(path)
    return {str(row.get("uid") or "") for row in rows if row.get("uid")}


def load_run(run_dir: Path) -> list[dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    results_candidates = [
        run_dir / "manual_results.jsonl",
        run_dir / "results_manual.jsonl",
    ]
    results_path = next((path for path in results_candidates if path.is_file()), None)
    if not manifest_path.is_file() or results_path is None:
        raise FileNotFoundError(f"{run_dir}: manifest/manual results missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("pillar") != "commentary":
        raise ValueError(f"{run_dir}: not a commentary manifest")
    items = {str(row.get("item_id") or ""): row for row in manifest.get("items") or []}
    results = read_jsonl(results_path)
    if not items or "" in items or len(items) != len(manifest.get("items") or []):
        raise ValueError(f"{run_dir}: invalid manifest item ids")
    output = []
    for result in results:
        item_id = str(result.get("item_id") or "")
        if item_id not in items:
            raise ValueError(f"{run_dir}: result {item_id} absent from manifest")
        item = items[item_id]
        expected_content_hash = str(item.get("content_manifest_sha256") or "")
        if expected_content_hash and result.get("content_manifest_sha256") != expected_content_hash:
            raise ValueError(f"{run_dir}: content lineage mismatch for {item_id}")
        decision = str(result.get("decision") or "")
        if decision not in ACCEPTED | {"reject"}:
            raise ValueError(f"{run_dir}: unresolved decision for {item_id}")
        output.append({
            "item_id": item_id,
            "uid": str(item.get("uid") or ""),
            "v1_decision": decision,
            "v1_normalized_behavior": str(result.get("normalized_behavior") or ""),
            "v1_normalized_norm": str(result.get("normalized_norm") or ""),
            "source_platform": str(item.get("source_platform") or "unknown"),
            "query_source": str(item.get("query_source") or "unknown"),
            "category": str(item.get("category") or "unknown"),
            "signal": str((item.get("detector_statement") or {}).get("signal") or item.get("polarity") or "unknown"),
            "source_manifest_path": str(manifest_path),
            "source_manifest_sha256": sha256(manifest_path),
            "source_results_path": str(results_path),
            "source_results_sha256": sha256(results_path),
            "content_manifest_sha256": str(item.get("content_manifest_sha256") or ""),
        })
    return output


def diverse_order(rows: Iterable[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["source_platform"], row["query_source"])].append(row)
    for key, items in groups.items():
        items.sort(key=lambda row: stable(seed, repr(key), row["uid"], row["item_id"]))
    keys = sorted(groups, key=lambda key: stable(seed, "group", repr(key)))
    output: list[dict[str, Any]] = []
    offset = 0
    while True:
        added = False
        for key in keys:
            if offset < len(groups[key]):
                output.append(groups[key][offset])
                added = True
        if not added:
            return output
        offset += 1


def select(
    rows: list[dict[str, Any]],
    excluded_uids: set[str],
    accepted_count: int,
    rejected_count: int,
    seed: str,
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["uid"] and row["uid"] not in excluded_uids]
    # A UID may occur in several historical audits.  Resolve that duplication
    # deterministically before balancing decisions, preserving source-disjointness.
    eligible.sort(key=lambda row: stable(seed, "dedup", row["uid"], row["item_id"], row["source_manifest_path"]))
    by_uid: dict[str, dict[str, Any]] = {}
    for row in eligible:
        by_uid.setdefault(row["uid"], row)
    unique = list(by_uid.values())
    prior_accepts = diverse_order(
        (row for row in unique if row["v1_decision"] in ACCEPTED), f"{seed}:accept"
    )
    prior_rejects = diverse_order(
        (row for row in unique if row["v1_decision"] == "reject"), f"{seed}:reject"
    )
    if len(prior_accepts) < accepted_count or len(prior_rejects) < rejected_count:
        raise ValueError(
            f"insufficient source-disjoint rows: accepts={len(prior_accepts)}, rejects={len(prior_rejects)}"
        )
    chosen = prior_accepts[:accepted_count] + prior_rejects[:rejected_count]
    chosen.sort(key=lambda row: stable(seed, "sealed", row["uid"], row["item_id"]))
    if len({row["uid"] for row in chosen}) != len(chosen):
        raise AssertionError("selected transfer cohort is not source-disjoint")
    return [
        {**row, "transfer_index": index, "manual_reaudit_required": True}
        for index, row in enumerate(chosen)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--exclude-gold", type=Path, required=True)
    parser.add_argument("--accepted-count", type=int, default=40)
    parser.add_argument("--rejected-count", type=int, default=20)
    parser.add_argument("--seed", default="commentary-occurred-event-v3-transfer")
    args = parser.parse_args()
    rows = []
    for run_dir in args.run_dir:
        rows.extend(load_run(run_dir))
    selected = select(
        rows,
        load_excluded_uids(args.exclude_gold),
        args.accepted_count,
        args.rejected_count,
        args.seed,
    )
    for row in selected:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
