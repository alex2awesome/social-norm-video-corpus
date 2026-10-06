#!/usr/bin/env python3
"""Compile a complete manual witnessed-v2 shadow review.

The output is deliberately not importable as a production keep decision.  It
records source recovery and instructional-reroute candidates for a subsequent
artifact-specific audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from witnessed_signal_contract import derive_witnessed_disposition


PROMPT_VERSION = "witnessed_atomic_sequence_v2"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: {exc}") from exc
    return rows


def compile_rows(
    manifest: dict[str, Any], manual_rows: list[dict[str, Any]], source_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = {item["item_id"]: item for item in manifest["items"]}
    manual = {row.get("item_id"): row for row in manual_rows}
    if None in manual or len(manual) != len(manual_rows):
        raise ValueError("manual rows require unique, nonempty item_id values")
    if set(manual) != set(items):
        missing = sorted(set(items) - set(manual))
        extra = sorted(set(manual) - set(items))
        raise ValueError(f"manual review must cover manifest exactly; missing={missing}, extra={extra}")

    now = time.time()
    records = []
    for item_id, item in items.items():
        raw = dict(manual[item_id])
        action_frames = raw.pop("action_evidence_frames", [])
        reaction_frames = raw.pop("reaction_evidence_frames", [])
        description = raw.pop("description")
        raw.pop("item_id", None)
        valid_frames = {frame["frame_index"] for frame in item.get("frames", [])}
        if not set(action_frames).issubset(valid_frames):
            raise ValueError(f"{item_id}: invalid action evidence frame")
        if not set(reaction_frames).issubset(valid_frames):
            raise ValueError(f"{item_id}: invalid reaction evidence frame")
        derived = derive_witnessed_disposition(raw)
        if derived["disposition"] in {"strict_accept", "recover_strict"}:
            duration = float(item["duration"])
            if float(derived["reaction_start_sec"]) >= duration:
                raise ValueError(f"{item_id}: reaction bound exceeds clip duration")
        records.append(
            {
                "item_id": item_id,
                "uid": item["uid"],
                "batch_id": manifest["batch_id"],
                "rubric_version": PROMPT_VERSION,
                "model": "codex-manual-exact-video-review",
                "frame_manifest_sha256": item["frame_manifest_sha256"],
                **derived,
                "action_evidence_frames": action_frames,
                "reaction_evidence_frames": reaction_frames,
                "description": description,
                "raw_response": {
                    "review_type": "manual atomic witnessed sequence audit",
                    "source_review": str(source_path),
                    "closed_model_api_call": False,
                    "corpus_mutated": False,
                },
                "audited_at": now,
            }
        )

    counts = Counter(record["disposition"] for record in records)
    summary = {
        "rubric_version": PROMPT_VERSION,
        "batch_id": manifest["batch_id"],
        "items": len(records),
        "source_disjoint": True,
        "complete_manual_coverage": len(records) == len(items),
        "dispositions": dict(sorted(counts.items())),
        "manual_review_sha256": file_sha256(source_path),
        "corpus_mutated": False,
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("manual_jsonl", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.summary.exists():
        raise SystemExit("refusing to overwrite existing output")
    manifest = json.loads(args.manifest.read_text())
    records, summary = compile_rows(manifest, load_jsonl(args.manual_jsonl), args.manual_jsonl)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

