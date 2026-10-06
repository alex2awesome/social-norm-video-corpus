#!/usr/bin/env python3
"""Freeze source-disjoint v8-positive and v8-reject instructional cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_instructional_vlm_discovery import v8_strict
    from scripts.select_instructional_v5_strict_candidates import latest_successes
else:
    from evaluate_instructional_vlm_discovery import v8_strict
    from select_instructional_v5_strict_candidates import latest_successes


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def balanced_source_disjoint(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    *,
    include_pattern: bool,
    used_uids: set[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["uid"]) in used_uids:
            continue
        key = (
            str(row.get("category") or "unknown"),
            str(row.get("polarity") or "unknown"),
        )
        if include_pattern:
            key += (str(row["v8_vote_pattern"]),)
        groups[key].append(row)
    for values in groups.values():
        values.sort(key=lambda row: stable_key(seed, str(row["item_id"])))
    keys = sorted(groups, key=lambda key: stable_key(seed, repr(key)))
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            values = groups[key]
            while values and str(values[0]["uid"]) in used_uids:
                values.pop(0)
            if not values:
                continue
            row = values.pop(0)
            used_uids.add(str(row["uid"]))
            selected.append(row)
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break
    return selected


def select(
    manifest: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    glm_rows: list[dict[str, Any]],
    excluded_uids: set[str],
    positive_count: int,
    reject_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    qwen = latest_successes(qwen_rows)
    glm = latest_successes(glm_rows)
    eligible: list[dict[str, Any]] = []
    for source in manifest:
        item_id = source["item_id"]
        if (
            item_id not in qwen
            or item_id not in glm
            or str(source["uid"]) in excluded_uids
        ):
            continue
        qp = v8_strict(qwen[item_id]["result"])
        gp = v8_strict(glm[item_id]["result"])
        row = dict(source)
        row["qwen_v8_result"] = qwen[item_id]["result"]
        row["glm_v8_result"] = glm[item_id]["result"]
        row["v8_vote_pattern"] = (
            "both_positive"
            if qp and gp
            else "qwen_only"
            if qp
            else "glm_only"
            if gp
            else "both_reject"
        )
        eligible.append(row)

    used = set(excluded_uids)
    positives = balanced_source_disjoint(
        [row for row in eligible if row["v8_vote_pattern"] == "both_positive"],
        positive_count,
        seed,
        include_pattern=False,
        used_uids=used,
    )
    rejects = balanced_source_disjoint(
        [row for row in eligible if row["v8_vote_pattern"] != "both_positive"],
        reject_count,
        seed + 1,
        include_pattern=True,
        used_uids=used,
    )
    selected: list[dict[str, Any]] = []
    for row in positives:
        row["audit_band"] = "v8_dual_strict"
        selected.append(row)
    for row in rejects:
        row["audit_band"] = "v8_conjunction_reject"
        selected.append(row)
    for index, row in enumerate(selected):
        row["audit_index"] = index
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--qwen-v8", required=True, type=Path)
    parser.add_argument("--glm-v8", required=True, type=Path)
    parser.add_argument("--excluded-uids", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--positive-count", type=int, default=60)
    parser.add_argument("--reject-count", type=int, default=30)
    args = parser.parse_args()
    if args.out.exists() or args.summary.exists():
        raise SystemExit("refusing to overwrite frozen validation artifacts")
    rows = select(
        load_jsonl(args.manifest),
        load_jsonl(args.qwen_v8),
        load_jsonl(args.glm_v8),
        set(args.excluded_uids.read_text().split()),
        args.positive_count,
        args.reject_count,
        args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    patterns: dict[str, int] = {}
    bands: dict[str, int] = {}
    for row in rows:
        patterns[row["v8_vote_pattern"]] = patterns.get(
            row["v8_vote_pattern"], 0
        ) + 1
        bands[row["audit_band"]] = bands.get(row["audit_band"], 0) + 1
    summary = {
        "kind": "instructional_v8_fresh_validation_selection",
        "seed": args.seed,
        "items": len(rows),
        "unique_uids": len({str(row["uid"]) for row in rows}),
        "by_band": dict(sorted(bands.items())),
        "by_v8_vote_pattern": dict(sorted(patterns.items())),
        "selection_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "corpus_action": "none_shadow_only",
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
