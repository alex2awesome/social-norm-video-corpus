#!/usr/bin/env python3
"""Export selected corpus metadata only after a blind rank audit is frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rows(
    corpus: list[dict[str, Any]],
    selection: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_item = {row["item_id"]: row for row in corpus}
    if len(by_item) != len(corpus):
        raise ValueError("duplicate corpus item_id")
    output = []
    for selected in sorted(
        selection, key=lambda row: int(row["audit_index"])
    ):
        item_id = selected["item_id"]
        if item_id not in by_item:
            raise ValueError(f"selected item absent from corpus: {item_id}")
        source = by_item[item_id]
        if source["uid"] != selected["uid"]:
            raise ValueError(f"uid mismatch for {item_id}")
        output.append(
            {
                **source,
                "audit_index": int(selected["audit_index"]),
            }
        )
    return output


def aligned_segments(
    transcript: dict[str, Any],
    start_sec: float,
    end_sec: float,
    tolerance: float = 0.5,
) -> list[dict[str, Any]]:
    output = []
    for segment in transcript.get("segments") or []:
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start_sec - tolerance or start > end_sec + tolerance:
            continue
        output.append(
            {
                "start": start,
                "end": end,
                "text": str(segment.get("text") or "").strip(),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--transcripts-dir", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    rows = select_rows(read_jsonl(args.manifest), read_jsonl(args.selection))
    transcript_coverage = 0
    if args.transcripts_dir:
        enriched = []
        for row in rows:
            transcript_path = args.transcripts_dir / f"{row['uid']}.json"
            aligned: list[dict[str, Any]] = []
            transcript_error = None
            try:
                transcript = json.loads(transcript_path.read_text())
                aligned = aligned_segments(
                    transcript,
                    float(row["start_sec"]),
                    float(row["end_sec"]),
                )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                transcript_error = f"{type(exc).__name__}: {exc}"
            if aligned:
                transcript_coverage += 1
            enriched.append(
                {
                    **row,
                    "aligned_transcript": aligned,
                    "transcript_error": transcript_error,
                }
            )
        rows = enriched
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "kind": "post_blind_rank_confirmation_semantics",
        "items": len(rows),
        "source_manifest_sha256": sha256(args.manifest),
        "selection_sha256": sha256(args.selection),
        "out_sha256": sha256(args.out),
        "aligned_transcript_items": transcript_coverage,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
