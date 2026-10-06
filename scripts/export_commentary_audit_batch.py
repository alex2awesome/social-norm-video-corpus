#!/usr/bin/env python3
"""Export a balanced, unjudged commentary batch with transcript context."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sqlite3
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from visual_audit_ledger import connect


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def select_balanced(
    conn: sqlite3.Connection,
    n: int,
    rubric: str,
    model: str,
    seed: str,
    query_sources: list[str],
    categories: list[str],
    uids: list[str],
    item_ids: list[str],
) -> list[sqlite3.Row]:
    clauses = [
        "i.pillar='commentary'",
        "i.present=1",
        """NOT EXISTS (
            SELECT 1
            FROM items prior_i
            JOIN commentary_judgments j ON j.item_id=prior_i.item_id
            WHERE prior_i.pillar=i.pillar AND prior_i.uid=i.uid
        )""",
        """NOT EXISTS (
            SELECT 1
            FROM items prior_i
            JOIN batch_items bi ON bi.item_id=prior_i.item_id
            WHERE prior_i.pillar=i.pillar AND prior_i.uid=i.uid
        )""",
    ]
    parameters: list[Any] = []
    for field, values in (
        ("query_source", query_sources),
        ("category", categories),
        ("uid", uids),
        ("item_id", item_ids),
    ):
        if values:
            clauses.append(f"i.{field} IN ({','.join('?' for _ in values)})")
            parameters.extend(values)
    rows = conn.execute(
        f"""
        SELECT i.* FROM items i
        WHERE {' AND '.join(clauses)}
        ORDER BY i.uid,i.item_index
        """,
        parameters,
    ).fetchall()
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["category"] or ""), str(row["polarity"] or ""))].append(row)
    queues: list[deque[sqlite3.Row]] = []
    for values in buckets.values():
        rng.shuffle(values)
        queues.append(deque(values))
    rng.shuffle(queues)
    selected: list[sqlite3.Row] = []
    while queues and len(selected) < n:
        next_round = []
        for queue in queues:
            if len(selected) >= n:
                break
            selected.append(queue.popleft())
            if queue:
                next_round.append(queue)
        queues = next_round
    return selected


def transcript_context(project_root: Path, row: sqlite3.Row, padding: float) -> list[dict[str, Any]]:
    transcript = load_json(project_root / "data" / "transcripts" / f"{row['uid']}.json")
    try:
        center_start = float(row["start_sec"])
        center_end = float(row["end_sec"])
    except (TypeError, ValueError):
        center_start = center_end = 0.0
    start = max(0.0, center_start - padding)
    end = center_end + padding
    context = []
    for segment in transcript.get("segments") or []:
        try:
            segment_start = float(segment["start"])
            segment_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if segment_end < start or segment_start > end:
            continue
        context.append(
            {
                "start": round(segment_start, 3),
                "end": round(segment_end, 3),
                "text": str(segment.get("text") or "").strip(),
                "overlaps_detector_span": not (
                    segment_end < center_start or segment_start > center_end
                ),
            }
        )
    return context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/visual_audit/audit.db"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--context-seconds", type=float, default=20.0)
    parser.add_argument("--rubric", default="commentary_v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--seed", default="commentary-full-audit-v1")
    parser.add_argument("--query-source", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--uid", action="append", default=[])
    parser.add_argument("--item-id", action="append", default=[])
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)
    project_root = args.project_root.resolve()
    conn = connect(args.db)
    selected = select_balanced(
        conn,
        args.count,
        args.rubric,
        args.model,
        args.seed,
        args.query_source,
        args.category,
        args.uid,
        args.item_id,
    )
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"comm_{stamp}_{hashlib.sha1(args.seed.encode()).hexdigest()[:8]}"
    manifest_path = args.out / "manifest.json"
    conn.execute(
        """
        INSERT INTO batches(
            batch_id,pillar,rubric_version,intended_model,selection_strategy,
            seed,manifest_path,created_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            batch_id, "commentary", args.rubric, args.model,
            "balanced_category_signal_unjudged_filtered", args.seed,
            str(manifest_path.resolve()), time.time(),
        ),
    )
    records = []
    for ordinal, row in enumerate(selected):
        metadata = load_json(project_root / row["metadata_path"])
        statements = metadata.get("statements") or []
        statement = statements[row["item_index"]] if row["item_index"] < len(statements) else {}
        record = {
            **dict(row),
            "ordinal": ordinal,
            "agent": metadata.get("agent") or (metadata.get("provenance") or {}).get("agent"),
            "detector_statement": statement,
            "transcript_context": transcript_context(project_root, row, args.context_seconds),
        }
        content_fields = {
            key: record.get(key)
            for key in (
                "item_id", "title", "category", "agent", "found_by_query", "query_source",
                "polarity", "norm", "start_quote", "start_sec", "end_sec",
                "detector_statement", "transcript_context",
            )
        }
        record["content_manifest_sha256"] = sha256_json(content_fields)
        records.append(record)
        conn.execute(
            """
            INSERT INTO batch_items(
                batch_id,item_id,ordinal,frame_count,frame_manifest_sha256
            ) VALUES (?,?,?,?,?)
            """,
            (batch_id, row["item_id"], ordinal, 0, record["content_manifest_sha256"]),
        )
    manifest = {
        "batch_id": batch_id,
        "pillar": "commentary",
        "rubric_version": args.rubric,
        "intended_model": args.model,
        "selection_strategy": "balanced_category_signal_unjudged_filtered",
        "filters": {
            "query_source": args.query_source,
            "category": args.category,
            "uid": args.uid,
            "item_id": args.item_id,
        },
        "seed": args.seed,
        "context_seconds": args.context_seconds,
        "items": records,
        "created_at": time.time(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with (args.out / "manual_review.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "ordinal", "item_id", "decision", "is_social_norm", "concrete_behavior",
                "social_actor_grounded", "target_or_shared_context_grounded",
                "normative_stance_grounded", "proposed_norm_supported", "stance_type",
                "quote_relation", "rejection_reasons", "behavior_evidence_quote",
                "stance_evidence_quote", "normalized_behavior", "normalized_norm",
                "manual_description",
            ]
        )
        for record in records:
            writer.writerow([record["ordinal"], record["item_id"]] + [""] * 15)
    conn.commit()
    conn.close()
    print(json.dumps({"batch_id": batch_id, "items": len(records), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
