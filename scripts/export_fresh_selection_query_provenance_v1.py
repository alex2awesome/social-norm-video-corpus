#!/usr/bin/env python3
"""Join sealed fresh instructional/witnessed cohorts to search provenance.

This reads only the exact selected UIDs from metadata and SQLite.  It preserves
conflicts rather than silently choosing one source, and creates no labels or
corpus dispositions.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def db_rows(path: Path, uids: set[str]) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(seen_videos)")
        }
        names = [
            name for name in (
                "video_id", "source", "platform", "title", "channel", "query",
                "query_source", "category", "status", "modality", "agent",
            ) if name in columns
        ]
        if "video_id" not in names:
            raise ValueError("seen_videos lacks video_id")
        output = {}
        ordered = sorted(uids)
        for offset in range(0, len(ordered), 400):
            batch = ordered[offset:offset + 400]
            placeholders = ",".join("?" for _ in batch)
            sql = (
                f"SELECT {','.join(names)} FROM seen_videos "
                f"WHERE video_id IN ({placeholders})"
            )
            for row in connection.execute(sql, batch):
                output[str(row["video_id"])] = dict(row)
        return output
    finally:
        connection.close()


def metadata_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def first(*values: Any) -> Any:
    return next((value for value in values if value not in (None, "")), None)


def source_record(
    pillar: str,
    selected: dict[str, Any],
    metadata: dict[str, Any],
    database: dict[str, Any],
) -> dict[str, Any]:
    provenance = metadata.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    query_values = {
        "selection": selected.get("found_by_query") or selected.get("query"),
        "metadata_provenance": provenance.get("found_by_query"),
        "metadata_top_level": metadata.get("found_by_query") or metadata.get("query"),
        "state_db": database.get("query"),
    }
    source_values = {
        "selection": selected.get("query_source"),
        "metadata_provenance": provenance.get("query_source"),
        "metadata_top_level": metadata.get("query_source"),
        "state_db": database.get("query_source"),
    }
    category_values = {
        "selection": selected.get("category"),
        "metadata_provenance": provenance.get("category"),
        "metadata_top_level": metadata.get("category"),
        "state_db": database.get("category"),
    }

    def conflict(values: dict[str, Any]) -> bool:
        observed = {str(value) for value in values.values() if value not in (None, "")}
        return len(observed) > 1

    return {
        "pillar": pillar,
        "item_id": str(selected.get("item_id") or ""),
        "uid": str(selected["uid"]),
        "cohort": selected.get("cohort"),
        "query": first(*query_values.values()),
        "query_source": first(*source_values.values()),
        "category": first(*category_values.values()),
        "platform": first(
            selected.get("source_platform"), selected.get("platform"),
            provenance.get("platform"), metadata.get("platform"),
            database.get("platform"), database.get("source"),
        ),
        "query_values_by_source": query_values,
        "query_source_values_by_source": source_values,
        "category_values_by_source": category_values,
        "query_conflict": conflict(query_values),
        "query_source_conflict": conflict(source_values),
        "category_conflict": conflict(category_values),
        "metadata_found": bool(metadata),
        "db_row_found": bool(database),
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def export(
    root: Path,
    instructional: list[dict[str, Any]],
    witnessed: list[dict[str, Any]],
    database: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for pillar, selected_rows, corpus_dir in (
        ("instructional", instructional, "instructional"),
        ("witnessed", witnessed, "hits"),
    ):
        for selected in selected_rows:
            uid = str(selected.get("uid") or "")
            key = (pillar, uid)
            if not uid or key in seen:
                raise ValueError(f"{pillar}: missing or duplicate selected uid")
            seen.add(key)
            metadata = metadata_row(
                root / "data" / corpus_dir / uid / "metadata.json"
            )
            row = source_record(pillar, selected, metadata, database.get(uid) or {})
            if not row["metadata_found"] and not row["db_row_found"]:
                raise ValueError(f"{pillar}:{uid}: no provenance source found")
            rows.append(row)
    summary = {
        "kind": "fresh_selection_query_provenance_v1",
        "items": len(rows),
        "by_pillar": {
            pillar: sum(row["pillar"] == pillar for row in rows)
            for pillar in ("instructional", "witnessed")
        },
        "query_present": sum(bool(row["query"]) for row in rows),
        "query_source_present": sum(bool(row["query_source"]) for row in rows),
        "query_conflicts": sum(row["query_conflict"] for row in rows),
        "query_source_conflicts": sum(row["query_source_conflict"] for row in rows),
        "read_only": True,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--instructional", type=Path, required=True)
    parser.add_argument("--witnessed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.summary.exists():
        raise FileExistsError("provenance output already exists")
    instruction = read_jsonl(args.instructional)
    witnessed = read_jsonl(args.witnessed)
    uids = {str(row["uid"]) for row in instruction + witnessed}
    rows, summary = export(
        args.root, instruction, witnessed,
        db_rows(args.root / "data/state.db", uids),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
