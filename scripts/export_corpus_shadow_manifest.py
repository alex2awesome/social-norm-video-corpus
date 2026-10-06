#!/usr/bin/env python3
"""Freeze a read-only corpus manifest for non-destructive shadow scoring.

The exporter reads the visual-audit inventory rather than corpus metadata
directly. Instructional and witnessed rows point at their already-cut clips.
Commentary rows point at retained source video and carry a bounded window around
the weak-label statement. It never changes the ledger, source metadata, clips,
or production routing state. Output paths must be new so every scoring run has
a stable denominator and identity hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ITEM_COLUMNS = (
    "item_id",
    "pillar",
    "uid",
    "item_index",
    "metadata_path",
    "clip_path",
    "title",
    "category",
    "genre",
    "source_platform",
    "found_by_query",
    "query_source",
    "polarity",
    "norm",
    "start_quote",
    "end_quote",
    "explanation",
    "start_sec",
    "end_sec",
    "source_mtime",
)


def sha256_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode())
    return digest.hexdigest()


def export_rows(
    conn: sqlite3.Connection,
    root: Path,
    pillar: str,
    limit: int | None = None,
    excluded_uids: set[str] | None = None,
    commentary_context_before: float = 12.0,
    commentary_context_after: float = 12.0,
) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    columns = ", ".join(ITEM_COLUMNS)
    media_condition = "" if pillar == "commentary" else "AND has_clip=1 "
    query = (
        f"SELECT {columns} FROM items "
        f"WHERE pillar=? AND present=1 {media_condition}"
        "ORDER BY uid,item_index"
    )
    parameters: list[Any] = [pillar]
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)
    records: list[dict[str, Any]] = []
    for source_row in conn.execute(query, parameters):
        row = dict(source_row)
        if excluded_uids and row["uid"] in excluded_uids:
            continue
        if pillar == "commentary":
            clip = root / "data" / "discussion_video" / f"{row['uid']}.mp4"
            try:
                statement_start = float(row.get("start_sec"))
                statement_end = float(row.get("end_sec"))
                media_start = max(
                    0.0, statement_start - commentary_context_before
                )
                media_end = max(
                    media_start,
                    statement_end + commentary_context_after,
                )
            except (TypeError, ValueError):
                media_start = None
                media_end = None
        else:
            clip = root / str(row["clip_path"])
            media_start = None
            media_end = None
        start = row.get("start_sec")
        end = row.get("end_sec")
        if media_start is not None and media_end is not None:
            duration = max(0.0, media_end - media_start)
        else:
            try:
                duration = max(0.0, float(end) - float(start))
            except (TypeError, ValueError):
                duration = 0.0
        row.update(
            {
                "ordinal": len(records),
                "source_clip": str(clip),
                "source_exists": clip.is_file(),
                "media_start_sec": media_start,
                "media_end_sec": media_end,
                "duration_hint": duration,
            }
        )
        records.append(row)
    return records


def source_disjoint_stratified_sample(
    records: list[dict[str, Any]], size: int, seed: str
) -> list[dict[str, Any]]:
    """Sample broadly across category/polarity while using each source once."""
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        cells[
            (str(row.get("polarity") or "none"), str(row.get("category") or "none"))
        ].append(row)
    for candidates in cells.values():
        rng.shuffle(candidates)
    keys = sorted(cells)
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    used_uids: set[str] = set()
    while len(selected) < size:
        progressed = False
        for key in keys:
            while cells[key] and cells[key][-1]["uid"] in used_uids:
                cells[key].pop()
            if not cells[key]:
                continue
            row = cells[key].pop()
            selected.append(row)
            used_uids.add(row["uid"])
            progressed = True
            if len(selected) >= size:
                break
        if not progressed:
            break
    selected.sort(key=lambda row: row["item_id"])
    for ordinal, row in enumerate(selected):
        row["ordinal"] = ordinal
    return selected


def judged_source_uids(
    conn: sqlite3.Connection, pillar: str, rubrics: list[str]
) -> set[str]:
    if not rubrics:
        return set()
    tables = {
        "instructional": "judgments",
        "witnessed": "witnessed_judgments",
        "commentary": "commentary_judgments",
    }
    placeholders = ",".join("?" for _ in rubrics)
    rows = conn.execute(
        f"""SELECT DISTINCT i.uid
            FROM {tables[pillar]} j JOIN items i USING(item_id)
            WHERE j.rubric_version IN ({placeholders})""",
        rubrics,
    )
    return {str(row[0]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--pillar",
        choices=("instructional", "witnessed", "commentary"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", default="corpus-shadow-manifest-v1")
    parser.add_argument("--commentary-context-before", type=float, default=12.0)
    parser.add_argument("--commentary-context-after", type=float, default=12.0)
    parser.add_argument(
        "--exclude-judged-rubric",
        action="append",
        default=[],
        help="Exclude every source UID already judged under this rubric.",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {args.out}")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.sample is not None and args.sample <= 0:
        raise SystemExit("--sample must be positive")
    if args.limit is not None and args.sample is not None:
        raise SystemExit("--limit and --sample are mutually exclusive")
    if (
        args.commentary_context_before < 0
        or args.commentary_context_after < 0
    ):
        raise SystemExit("commentary context must be non-negative")

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    excluded_uids = judged_source_uids(
        conn, args.pillar, args.exclude_judged_rubric
    )
    records = export_rows(
        conn,
        args.root.resolve(),
        args.pillar,
        args.limit,
        excluded_uids=excluded_uids,
        commentary_context_before=args.commentary_context_before,
        commentary_context_after=args.commentary_context_after,
    )
    if args.sample is not None:
        records = source_disjoint_stratified_sample(records, args.sample, args.seed)
    lines = [json.dumps(row, sort_keys=True) + "\n" for row in records]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines))

    summary_path = args.summary or args.out.with_suffix(".summary.json")
    payload = {
        "schema_version": 1,
        "kind": "corpus_shadow_manifest",
        "pillar": args.pillar,
        "records": len(records),
        "unique_sources": len({row["uid"] for row in records}),
        "missing_source_clips": sum(not row["source_exists"] for row in records),
        "excluded_judged_source_uids": len(excluded_uids),
        "seed": args.seed if args.sample is not None else None,
        "commentary_context_before": (
            args.commentary_context_before
            if args.pillar == "commentary"
            else None
        ),
        "commentary_context_after": (
            args.commentary_context_after
            if args.pillar == "commentary"
            else None
        ),
        "manifest_sha256": sha256_lines(lines),
        "manifest": str(args.out),
        "corpus_mutated": False,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
