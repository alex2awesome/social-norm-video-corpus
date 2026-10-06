#!/usr/bin/env python3
"""Freeze a source-disjoint prospective instructional source population."""

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


def read_exclusions(
    uid_files: list[Path],
    source_manifests: list[Path],
) -> set[str]:
    excluded: set[str] = set()
    for path in uid_files:
        excluded.update(line.strip() for line in path.read_text().splitlines() if line.strip())
    for path in source_manifests:
        for line in path.read_text().splitlines():
            if line.strip():
                excluded.add(str(json.loads(line)["uid"]))
    return excluded


def candidates(
    instructional_root: Path,
    excluded_uids: set[str],
    seed: str,
    platform: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(
        instructional_root.glob(f"{platform}__*/metadata.json")
    ):
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
        eligible.sort(key=lambda value: rank(seed, f"{uid}:{value[0]}"))
        demo_index, demo, clip_path = eligible[0]
        rows.append(
            {
                "item_id": f"instructional:{uid}:{demo_index}",
                "uid": uid,
                "pillar": "instructional",
                "source_platform": platform,
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


def freeze(
    instructional_root: Path,
    excluded_uids: set[str],
    seed: str,
    platform: str,
    size: int | None,
) -> list[dict[str, Any]]:
    population = candidates(
        instructional_root,
        excluded_uids,
        seed,
        platform,
    )
    if size is not None:
        if size <= 0:
            raise ValueError("size must be positive")
        if len(population) < size:
            raise ValueError(
                f"requested {size} sources but only {len(population)} are eligible"
            )
        population = population[:size]
    for prospective_index, row in enumerate(population):
        row["prospective_index"] = prospective_index
        row["source_clip_sha256"] = sha256(Path(row["source_clip"]))
    return population


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instructional-root", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path, action="append", default=[])
    parser.add_argument(
        "--exclude-source-manifest",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--seed", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--size", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    excluded = read_exclusions(
        args.exclude_uids,
        args.exclude_source_manifest,
    )
    rows = freeze(
        args.instructional_root,
        excluded,
        args.seed,
        args.platform,
        args.size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary = {
        "kind": "instructional_v15_prospective_source_population",
        "items": len(rows),
        "distinct_uids": len({row["uid"] for row in rows}),
        "source_platform": args.platform,
        "polarity": "violation",
        "complete_eligible_population": args.size is None,
        "requested_size": args.size,
        "seed": args.seed,
        "excluded_uids": len(excluded),
        "selection_sha256": sha256(args.out),
        "exclusion_artifact_sha256": {
            str(path): sha256(path)
            for path in [*args.exclude_uids, *args.exclude_source_manifest]
        },
        "corpus_mutated": False,
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

