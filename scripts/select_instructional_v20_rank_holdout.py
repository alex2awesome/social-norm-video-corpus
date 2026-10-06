#!/usr/bin/env python3
"""Freeze a blind V20 holdout across high, boundary, and low visual ranks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(seed: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def select(
    rows: list[dict[str, Any]],
    high_count: int,
    boundary_count: int,
    low_count: int,
    seed: str,
    excluded_uids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded_uids = excluded_uids or set()
    eligible = [
        row for row in rows if str(row.get("uid") or "") not in excluded_uids
    ]
    high = [row for row in eligible if row["development_review_band"]]
    below = [row for row in eligible if not row["development_review_band"]]
    if len(high) < high_count or len(below) < boundary_count + low_count:
        raise ValueError("insufficient rows for requested bands")
    used_uids: set[str] = set()

    def source_disjoint_take(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        chosen = []
        for row in candidates:
            uid = str(row.get("uid") or row["item_id"])
            if uid in used_uids:
                continue
            chosen.append(row)
            used_uids.add(uid)
            if len(chosen) == count:
                break
        return chosen

    chosen_high = source_disjoint_take(sorted(
        high, key=lambda row: stable_key(seed, str(row["item_id"]))
    ), high_count)
    chosen_boundary = source_disjoint_take(sorted(
        below,
        key=lambda row: (
            -float(row["visual_rank_score"]),
            stable_key(seed, str(row["item_id"])),
        ),
    ), boundary_count)
    boundary_ids = {str(row["item_id"]) for row in chosen_boundary}
    chosen_low = source_disjoint_take(sorted(
        (row for row in below if str(row["item_id"]) not in boundary_ids),
        key=lambda row: (
            float(row["visual_rank_score"]),
            stable_key(seed, str(row["item_id"])),
        ),
    ), low_count)
    if tuple(map(len, (chosen_high, chosen_boundary, chosen_low))) != (
        high_count,
        boundary_count,
        low_count,
    ):
        raise ValueError("insufficient source-disjoint rows for requested bands")
    selected = [
        *({**row, "sealed_rank_band": "high"} for row in chosen_high),
        *(
            {**row, "sealed_rank_band": "boundary_below"}
            for row in chosen_boundary
        ),
        *({**row, "sealed_rank_band": "low"} for row in chosen_low),
    ]
    return sorted(
        selected,
        key=lambda row: stable_key(f"{seed}:blind", str(row["item_id"])),
    )


def blind_record(row: dict[str, Any], audit_index: int, candidate_id: str) -> dict[str, Any]:
    """Emit only identity and media fields; old manifests may lack clip hashes."""
    record = {
        "audit_index": audit_index,
        "candidate_id": candidate_id,
        "item_id": row["item_id"],
        "uid": row["uid"],
        "pillar": row.get("pillar") or "instructional",
        "source_clip": row["source_clip"],
    }
    if row.get("source_sha256"):
        record["source_sha256"] = row["source_sha256"]
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--high", type=int, default=30)
    parser.add_argument("--boundary", type=int, default=15)
    parser.add_argument("--low", type=int, default=15)
    parser.add_argument("--seed", default="instructional-v20-rank-holdout-v1")
    parser.add_argument("--exclude-uids", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    selected = select(
        read_jsonl(args.ranking),
        args.high,
        args.boundary,
        args.low,
        args.seed,
        {
            line.strip()
            for line in args.exclude_uids.read_text().splitlines()
            if line.strip()
        }
        if args.exclude_uids
        else set(),
    )
    args.out.mkdir(parents=True)
    semantic = []
    blind = []
    for audit_index, row in enumerate(selected):
        candidate_id = f"v20holdout-{audit_index:04d}"
        semantic.append(
            {
                **row,
                "audit_index": audit_index,
                "candidate_id": candidate_id,
            }
        )
        blind.append(blind_record(row, audit_index, candidate_id))
    semantic_path = args.out / "sealed_selection.jsonl"
    blind_path = args.out / "blind_source_manifest.jsonl"
    semantic_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in semantic)
    )
    blind_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind)
    )
    summary = {
        "kind": "instructional_v20_rank_source_disjoint_holdout",
        "status": "preregistered_blind_manual_audit_pending",
        "policy": "shadow_only_no_keep_reject_or_corpus_mutation",
        "seed": args.seed,
        "ranking_rows": len(read_jsonl(args.ranking)),
        "selected": len(selected),
        "bands": {
            "high": args.high,
            "boundary_below": args.boundary,
            "low": args.low,
        },
        "semantic_fields_in_blind_manifest": False,
        "excluded_uids": (
            sum(bool(line.strip()) for line in args.exclude_uids.read_text().splitlines())
            if args.exclude_uids
            else 0
        ),
        "artifact_sha256": {
            "ranking": sha256(args.ranking),
            "sealed_selection": sha256(semantic_path),
            "blind_source_manifest": sha256(blind_path),
        },
    }
    (args.out / "preregistration.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
