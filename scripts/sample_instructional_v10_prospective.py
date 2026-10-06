#!/usr/bin/env python3
"""Freeze a source-disjoint prospective sample of YouTube violation demos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def candidates(
    instructional_root: Path,
    excluded_uids: set[str],
    seed: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(instructional_root.glob("youtube__*/metadata.json")):
        uid = metadata_path.parent.name
        if uid in excluded_uids:
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        eligible: list[tuple[int, dict[str, Any], Path]] = []
        for demo_index, demo in enumerate(metadata.get("demos") or []):
            clip_name = demo.get("clip")
            clip_path = metadata_path.parent / str(clip_name or "")
            if (
                demo.get("polarity") == "violation"
                and clip_name
                and clip_path.is_file()
                and clip_path.stat().st_size > 0
            ):
                eligible.append((demo_index, demo, clip_path))
        if not eligible:
            continue
        # One clip per source. The per-source choice and global ordering are
        # deterministic and do not use title, category, norm, or model output.
        eligible.sort(
            key=lambda value: rank(seed, f"{uid}:{value[0]}")
        )
        demo_index, demo, clip_path = eligible[0]
        item_id = f"instructional:{uid}:{demo_index}"
        rows.append(
            {
                "item_id": item_id,
                "uid": uid,
                "pillar": "instructional",
                "source_platform": "youtube",
                "source_clip": str(clip_path.resolve()),
                "title": metadata.get("title"),
                "category": metadata.get("category")
                or (metadata.get("provenance") or {}).get("category"),
                "norm": demo.get("norm"),
                "polarity": demo.get("polarity"),
                "start_quote": demo.get("start_quote"),
                "end_quote": demo.get("end_quote"),
                "explanation": demo.get("explanation"),
                "demo_index": demo_index,
            }
        )
    rows.sort(key=lambda row: rank(seed, row["item_id"]))
    return rows


def sample(
    instructional_root: Path,
    exclusion_path: Path,
    seed: str,
    size: int,
) -> list[dict[str, Any]]:
    if size <= 0:
        raise ValueError("size must be positive")
    excluded = {
        line.strip()
        for line in exclusion_path.read_text().splitlines()
        if line.strip()
    }
    population = candidates(instructional_root, excluded, seed)
    if len(population) < size:
        raise ValueError(
            f"requested {size} sources but only {len(population)} are eligible"
        )
    selected = population[:size]
    for audit_index, row in enumerate(selected):
        row["prospective_index"] = audit_index
        row["source_clip_sha256"] = sha256(Path(row["source_clip"]))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instructional-root", type=Path, required=True)
    parser.add_argument("--excluded-uids", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    rows = sample(
        args.instructional_root,
        args.excluded_uids,
        args.seed,
        args.size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary = {
        "items": len(rows),
        "distinct_uids": len({row["uid"] for row in rows}),
        "source_platforms": sorted({row["source_platform"] for row in rows}),
        "polarities": sorted({row["polarity"] for row in rows}),
        "seed": args.seed,
        "selection_sha256": sha256(args.out),
        "exclusion_sha256": sha256(args.excluded_uids),
        "corpus_mutated": False,
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
