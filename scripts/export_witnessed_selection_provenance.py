#!/usr/bin/env python3
"""Export compact search provenance for a sealed witnessed audit selection."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fetch_rows(db_path: Path, uids: set[str]) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(seen_videos)")
        }
        wanted = [
            name for name in (
                "video_id", "source", "platform", "title", "channel", "query",
                "query_source", "category", "status", "modality", "agent",
            ) if name in columns
        ]
        if "video_id" not in wanted:
            raise ValueError("seen_videos lacks video_id")
        result: dict[str, dict[str, Any]] = {}
        ordered = sorted(uids)
        for start in range(0, len(ordered), 400):
            batch = ordered[start : start + 400]
            placeholders = ",".join("?" for _ in batch)
            query = (
                f"SELECT {','.join(wanted)} FROM seen_videos "
                f"WHERE video_id IN ({placeholders})"
            )
            for row in connection.execute(query, batch):
                result[str(row["video_id"])] = dict(row)
        return result
    finally:
        connection.close()


def export(
    selection: list[dict[str, Any]],
    db_rows: dict[str, dict[str, Any]],
    staging_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    uids = [str(row["uid"]) for row in selection]
    if len(uids) != len(set(uids)):
        raise ValueError("selection must be source-disjoint")
    staging = {str(row["uid"]): row for row in staging_rows}
    if len(staging) != len(staging_rows):
        raise ValueError("duplicate staging UID")
    rows = []
    for selected in selection:
        uid = str(selected["uid"])
        db = db_rows.get(uid) or {}
        score = staging.get(uid) or {}
        rows.append({
            "item_id": str(selected["item_id"]),
            "uid": uid,
            "cohort": str(selected["cohort"]),
            "platform": selected.get("platform") or db.get("platform") or db.get("source"),
            "title": db.get("title"),
            "channel": db.get("channel"),
            "query": db.get("query"),
            "query_source": db.get("query_source"),
            "category": db.get("category"),
            "status": db.get("status"),
            "modality": db.get("modality"),
            "agent": db.get("agent"),
            "db_row_found": uid in db_rows,
            "title_staging_cue": score.get("title_staging_cue"),
            "title_creator_initiated_candidate_cue": score.get(
                "title_creator_initiated_candidate_cue"
            ),
            "transcript_explicit_reveal_cue": score.get(
                "transcript_explicit_reveal_cue"
            ),
            "transcript_creator_setup_cue": score.get(
                "transcript_creator_setup_cue"
            ),
            "staging_score_error": score.get("error"),
            "policy": "compact_read_only_search_provenance_audit",
        })
    summary = {
        "kind": "witnessed_sealed_selection_search_provenance_v1",
        "items": len(rows),
        "unique_sources": len(set(uids)),
        "db_rows_found": sum(row["db_row_found"] for row in rows),
        "staging_rows_found": sum(row["uid"] in staging for row in rows),
        "source_disjoint": True,
        "corpus_mutated": False,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--staging-scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.summary.exists():
        raise SystemExit("refusing to overwrite existing provenance artifacts")
    selection = read_jsonl(args.selection)
    rows, summary = export(
        selection,
        fetch_rows(args.state_db, {str(row["uid"]) for row in selection}),
        read_jsonl(args.staging_scores),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

