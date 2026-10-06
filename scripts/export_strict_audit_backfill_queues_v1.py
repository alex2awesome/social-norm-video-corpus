#!/usr/bin/env python3
"""Export resumable scoring queues from a frozen strict-audit coverage ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scoring_row(row: dict[str, Any], root: Path) -> dict[str, Any]:
    start = row.get("start_sec")
    end = row.get("end_sec")
    if row["pillar"] == "commentary":
        try:
            media_start = max(0.0, float(start) - 12.0)
            media_end = max(media_start, float(end) + 12.0)
        except (TypeError, ValueError):
            media_start = None
            media_end = None
    else:
        # Instructional and witnessed inputs are already cut clips. Their
        # source-video timestamps must not be applied a second time.
        media_start = None
        media_end = None
    return {
        "item_id": row["item_id"],
        "pillar": row["pillar"],
        "uid": row["uid"],
        "source_clip": str((root / row["media_path"]).resolve()),
        "media_start_sec": media_start,
        "media_end_sec": media_end,
        "norm": row.get("norm"),
        "polarity": row.get("polarity") or row.get("reaction_tag") or row.get("signal"),
        "coverage_manifest_item_id": row["item_id"],
        "automatic_acceptance": False,
        "automatic_rejection": False,
    }


def export(
    coverage_manifest: Path,
    root: Path,
    out: Path,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite frozen queue directory: {out}")
    rows = list(read_jsonl(coverage_manifest))
    low_level = [
        scoring_row(row, root)
        for row in rows
        if row.get("media_present") and not row.get("low_level_visual_complete")
    ]
    strict = [
        scoring_row(row, root)
        for row in rows
        if row.get("media_present") and row.get("strict_audit_needed")
    ]
    out.mkdir(parents=True)
    low_path = out / "low_level_delta_manifest.jsonl"
    strict_path = out / "strict_vlm_manifest.jsonl"
    low_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in low_level)
    )
    strict_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in strict)
    )
    summary = {
        "schema_version": 1,
        "kind": "strict_audit_backfill_queues_v1",
        "policy": "shadow_only_no_accept_reject_or_delete",
        "coverage_manifest": str(coverage_manifest.resolve()),
        "coverage_manifest_sha256": sha256(coverage_manifest),
        "low_level_delta_items": len(low_level),
        "low_level_delta_by_pillar": dict(
            sorted(Counter(row["pillar"] for row in low_level).items())
        ),
        "strict_vlm_items": len(strict),
        "strict_vlm_by_pillar": dict(
            sorted(Counter(row["pillar"] for row in strict).items())
        ),
        "low_level_manifest_sha256": sha256(low_path),
        "strict_vlm_manifest_sha256": sha256(strict_path),
        "corpus_mutated": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = export(args.coverage_manifest, args.root.resolve(), args.out)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
