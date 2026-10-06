#!/usr/bin/env python3
"""Package selected source clips without exposing sealed semantic metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


BLIND_FIELDS = {
    "audit_index",
    "candidate_id",
    "sheet_path",
    "sheet_sha256",
}
OUTPUT_FIELDS = {
    "audit_index",
    "candidate_id",
    "clip_path",
    "clip_sha256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def package(
    blind_manifest: Path,
    semantic_manifest: Path,
    output_dir: Path,
    audit_indices: set[int],
) -> list[dict[str, Any]]:
    blind = read_jsonl(blind_manifest)
    semantic = read_jsonl(semantic_manifest)
    if not blind or not semantic:
        raise ValueError("manifests must be non-empty")
    if any(set(row) != BLIND_FIELDS for row in blind):
        raise ValueError("blind manifest contains missing or semantic fields")
    if len({int(row["audit_index"]) for row in blind}) != len(blind):
        raise ValueError("blind manifest contains duplicate audit_index")
    if len({str(row["candidate_id"]) for row in blind}) != len(blind):
        raise ValueError("blind manifest contains duplicate candidate_id")

    blind_by_index = {int(row["audit_index"]): row for row in blind}
    unknown = audit_indices - set(blind_by_index)
    if unknown:
        raise ValueError(f"unknown audit indices: {sorted(unknown)}")
    semantic_by_candidate: dict[str, dict[str, Any]] = {}
    for row in semantic:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in semantic_by_candidate:
            raise ValueError("semantic manifest has missing or duplicate candidate_id")
        semantic_by_candidate[candidate_id] = row
    if set(semantic_by_candidate) != {
        str(row["candidate_id"]) for row in blind
    }:
        raise ValueError("blind and semantic candidate coverage differs")

    output_dir.mkdir(parents=True, exist_ok=True)
    packaged: list[dict[str, Any]] = []
    for audit_index in sorted(audit_indices):
        blind_row = blind_by_index[audit_index]
        candidate_id = str(blind_row["candidate_id"])
        sealed = semantic_by_candidate[candidate_id]
        source = Path(str(sealed["source_clip"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = sha256(source)
        expected = sealed.get("source_clip_sha256")
        if expected and actual != expected:
            raise ValueError(
                f"audit_index {audit_index}: sealed source clip hash mismatch"
            )
        suffix = source.suffix.lower() or ".mp4"
        target = output_dir / f"{candidate_id}{suffix}"
        if target.exists():
            if sha256(target) != actual:
                raise ValueError(f"refusing to replace mismatched {target}")
        else:
            shutil.copy2(source, target)
        row = {
            "audit_index": audit_index,
            "candidate_id": candidate_id,
            "clip_path": str(target.resolve()),
            "clip_sha256": actual,
        }
        if set(row) != OUTPUT_FIELDS:
            raise AssertionError("output field allowlist violated")
        packaged.append(row)
    return packaged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--audit-index", type=int, action="append", required=True)
    args = parser.parse_args()
    if args.output_manifest.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_manifest}")
    rows = package(
        args.blind_manifest,
        args.semantic_manifest,
        args.output_dir,
        set(args.audit_index),
    )
    args.output_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary = {
        "kind": "blind_clip_package",
        "items": len(rows),
        "audit_indices": sorted(set(args.audit_index)),
        "semantic_metadata_emitted": False,
        "blind_manifest_sha256": sha256(args.blind_manifest),
        "semantic_manifest_sha256": sha256(args.semantic_manifest),
        "output_manifest_sha256": sha256(args.output_manifest),
    }
    args.output_manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
