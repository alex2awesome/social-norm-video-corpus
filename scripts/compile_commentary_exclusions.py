#!/usr/bin/env python3
"""Compile source-level commentary exclusions from legacy lists and manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.sample_commentary_visual_audit import normalize_source_uid
else:
    from sample_commentary_visual_audit import normalize_source_uid


PLATFORMS = ("dailymotion__", "youtube__", "rumble__", "reddit__")


def source_uid(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = normalize_source_uid(value)
    return normalized if normalized.startswith(PLATFORMS) else None


def collect_json_uids(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"uid", "video_id", "item_id", "parent_item_id"}:
                normalized = source_uid(nested)
                if normalized:
                    output.add(normalized)
            output.update(collect_json_uids(nested))
    elif isinstance(value, list):
        for nested in value:
            output.update(collect_json_uids(nested))
    return output


def load_manifest(path: Path) -> set[str]:
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        return collect_json_uids(rows)
    return collect_json_uids(json.loads(path.read_text()))


def compile_exclusions(
    base_paths: list[Path],
    manifest_paths: list[Path],
) -> list[str]:
    output: set[str] = set()
    for path in base_paths:
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            normalized = source_uid(line)
            if normalized:
                output.add(normalized)
    for path in manifest_paths:
        output.update(load_manifest(path))
    return sorted(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    args = parser.parse_args()
    rows = compile_exclusions(args.base, args.manifest)
    payload = "".join(f"{uid}\n" for uid in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload)
    metadata = {
        "schema_version": 1,
        "kind": "source_level_commentary_exclusions",
        "base_paths": [str(path) for path in args.base],
        "manifest_paths": [str(path) for path in args.manifest],
        "source_uids": len(rows),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    args.metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
