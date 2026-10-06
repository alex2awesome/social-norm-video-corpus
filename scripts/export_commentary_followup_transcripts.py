#!/usr/bin/env python3
"""Export timestamped transcript evidence for unresolved commentary audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_ledger(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_uids(
    semantic_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    semantic = {row["candidate_id"]: row for row in semantic_rows}
    if len(semantic) != len(semantic_rows):
        raise ValueError("duplicate semantic candidate_id")
    output = []
    for ledger in ledger_rows:
        if ledger["visual_status"] != "U":
            continue
        if ledger["candidate_id"] not in semantic:
            raise ValueError(f"missing semantic row: {ledger['candidate_id']}")
        output.append(semantic[ledger["candidate_id"]])
    return sorted(output, key=lambda row: int(row["audit_index"]))


def transcript_record(row: dict[str, Any], transcript_root: Path) -> dict[str, Any]:
    path = transcript_root / f"{row['uid']}.json"
    base = {
        "audit_index": row["audit_index"],
        "candidate_id": row["candidate_id"],
        "uid": row["uid"],
        "title": row["norm"],
    }
    if not path.is_file():
        return {**base, "transcript_status": "missing", "segments": []}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **base,
            "transcript_status": "error",
            "transcript_error": f"{type(exc).__name__}: {exc}",
            "segments": [],
        }
    segments = []
    for segment in payload.get("segments") or []:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": text,
            }
        )
    return {
        **base,
        "transcript_status": "available",
        "language": payload.get("language"),
        "segments": segments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--blind-ledger", type=Path, required=True)
    parser.add_argument("--transcript-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    selected = select_uids(
        load_jsonl(args.semantic_manifest),
        load_ledger(args.blind_ledger),
    )
    records = [
        transcript_record(row, args.transcript_root)
        for row in selected
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    summary = {
        "kind": "commentary_unresolved_transcript_followup_v1",
        "selected_unresolved": len(records),
        "available_transcripts": sum(
            row["transcript_status"] == "available" for row in records
        ),
        "missing_transcripts": sum(
            row["transcript_status"] == "missing" for row in records
        ),
        "errored_transcripts": sum(
            row["transcript_status"] == "error" for row in records
        ),
        "semantic_manifest_sha256": sha256(args.semantic_manifest),
        "blind_ledger_sha256": sha256(args.blind_ledger),
        "output_sha256": sha256(args.out),
        "corpus_mutated": False,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not summary["errored_transcripts"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
