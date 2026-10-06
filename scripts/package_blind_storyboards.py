#!/usr/bin/env python3
"""Copy a frozen blind storyboard audit without leaking semantic metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ALLOWED_FIELDS = {
    "audit_index",
    "candidate_id",
    "sheet_path",
    "sheet_sha256",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(
    manifest: Path,
    output_dir: Path,
    source_root: Path | None = None,
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in manifest.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("blind manifest is empty")
    if any(set(row) != ALLOWED_FIELDS for row in rows):
        raise ValueError("blind manifest contains missing or semantic fields")
    expected = set(range(len(rows)))
    observed = {int(row["audit_index"]) for row in rows}
    if observed != expected:
        raise ValueError("blind manifest audit_index coverage is not contiguous")
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("blind manifest contains duplicate candidate_id")

    output_dir.mkdir(parents=True, exist_ok=True)
    packaged: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: int(value["audit_index"])):
        source = Path(row["sheet_path"])
        if not source.is_absolute():
            source = (source_root or manifest.parent) / source
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = sha256(source)
        if actual != row["sheet_sha256"]:
            raise ValueError(
                f"audit_index {row['audit_index']}: storyboard hash mismatch"
            )
        target = output_dir / f"{row['candidate_id']}.jpg"
        if target.exists():
            if sha256(target) != actual:
                raise ValueError(f"refusing to replace mismatched {target}")
        else:
            shutil.copy2(source, target)
        packaged.append(
            {
                "audit_index": int(row["audit_index"]),
                "candidate_id": row["candidate_id"],
                "sheet_path": str(target.resolve()),
                "sheet_sha256": actual,
            }
        )
    return packaged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "base directory for relative sheet_path values; defaults to the "
            "blind manifest directory"
        ),
    )
    args = parser.parse_args()
    if args.output_manifest.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_manifest}")
    rows = package(args.manifest, args.output_dir, args.source_root)
    args.output_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary = {
        "kind": "blind_storyboard_package",
        "items": len(rows),
        "semantic_metadata_emitted": False,
        "source_manifest_sha256": sha256(args.manifest),
        "output_manifest_sha256": sha256(args.output_manifest),
    }
    args.output_manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
