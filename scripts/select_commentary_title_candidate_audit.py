#!/usr/bin/env python3
"""Freeze a source-disjoint stratified audit of commentary title candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_key(seed: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}\0{row['item_id']}".encode()).hexdigest()


def select(
    rows: list[dict[str, Any]],
    count: int,
    seed: str,
    excluded_uids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded_uids = excluded_uids or set()
    item_ids = [str(row.get("item_id") or "") for row in rows]
    if any(not value for value in item_ids) or len(set(item_ids)) != len(item_ids):
        raise ValueError("candidate rows have missing or duplicate item IDs")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        uid = str(row.get("uid") or "")
        if not uid or uid in excluded_uids:
            continue
        cues = tuple(sorted(map(str, row.get("title_event_cues") or [])))
        groups[cues or ("<no-cue>",)].append(row)
    for candidates in groups.values():
        candidates.sort(key=lambda row: stable_key(seed, row))
    strata = sorted(groups)
    selected: list[dict[str, Any]] = []
    used_uids: set[str] = set()
    while strata and len(selected) < count:
        remaining = []
        for stratum in strata:
            candidates = groups[stratum]
            row = None
            while candidates:
                candidate = candidates.pop(0)
                if str(candidate["uid"]) not in used_uids:
                    row = candidate
                    break
            if row is not None:
                selected.append(row)
                used_uids.add(str(row["uid"]))
            if candidates:
                remaining.append(stratum)
            if len(selected) == count:
                break
        strata = remaining
    if len(selected) != count:
        raise ValueError(f"wanted {count} source-disjoint rows, found {len(selected)}")
    selected.sort(key=lambda row: stable_key(f"{seed}:blind", row))
    return [
        {**row, "audit_index": index, "candidate_id": f"commentary-expansion-{index:04d}"}
        for index, row in enumerate(selected)
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", default="commentary-title-expansion-audit-v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    excluded = (
        {line.strip() for line in args.exclude_uids.read_text().splitlines() if line.strip()}
        if args.exclude_uids
        else set()
    )
    rows = select(read_jsonl(args.candidates), args.count, args.seed, excluded)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = {
        "kind": "commentary_title_candidate_source_disjoint_audit_selection",
        "policy": "blind_manual_audit_shadow_only_no_corpus_mutation",
        "selected": len(rows),
        "unique_uids": len({row["uid"] for row in rows}),
        "excluded_uids": len(excluded),
        "seed": args.seed,
        "source_sha256": sha256(args.candidates),
        "selection_sha256": sha256(args.out),
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
