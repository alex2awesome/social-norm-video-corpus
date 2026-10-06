#!/usr/bin/env python3
"""Export a balanced, never-before-judged audit batch with temporal frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sqlite3
import subprocess
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .visual_audit_ledger import connect
except ImportError:  # Direct script execution.
    from visual_audit_ledger import connect


def probe_duration(ffprobe: str, clip: Path) -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip)],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError(f"invalid duration for {clip}")
    return duration


def select_balanced(
    conn: sqlite3.Connection,
    pillar: str,
    n: int,
    rubric: str,
    model: str,
    seed: str,
    polarities: list[str],
    categories: list[str],
    query_regex: str,
    query_sources: list[str],
    uids: list[str],
    item_ids: list[str],
) -> list[sqlite3.Row]:
    judgment_table = "witnessed_judgments" if pillar == "witnessed" else "judgments"
    clauses = [
        "i.pillar=?",
        "i.present=1",
        "i.has_clip=1",
        f"""NOT EXISTS (
            SELECT 1
            FROM items prior_i
            JOIN {judgment_table} j ON j.item_id=prior_i.item_id
            WHERE prior_i.pillar=i.pillar AND prior_i.uid=i.uid
        )""",
    ]
    # An explicit frozen item list is also the escalation path from a lightweight
    # hourly batch to a denser review of those exact same unjudged items.  Do not
    # let that earlier batch reservation hide them; prior judgments still exclude.
    if not item_ids:
        clauses.append(
            """NOT EXISTS (
                SELECT 1
                FROM items prior_i
                JOIN batch_items bi ON bi.item_id=prior_i.item_id
                WHERE prior_i.pillar=i.pillar AND prior_i.uid=i.uid
            )"""
        )
    # This exporter promises a never-before-judged sample.  Cross-model or
    # cross-rubric rechecks must use an explicit item-id manifest instead of
    # silently contaminating fresh precision cohorts.
    parameters: list[Any] = [pillar]
    if polarities:
        clauses.append(f"i.polarity IN ({','.join('?' for _ in polarities)})")
        parameters.extend(polarities)
    if categories:
        clauses.append(f"i.category IN ({','.join('?' for _ in categories)})")
        parameters.extend(categories)
    if uids:
        clauses.append(f"i.uid IN ({','.join('?' for _ in uids)})")
        parameters.extend(uids)
    if query_sources:
        clauses.append(f"i.query_source IN ({','.join('?' for _ in query_sources)})")
        parameters.extend(query_sources)
    if item_ids:
        clauses.append(f"i.item_id IN ({','.join('?' for _ in item_ids)})")
        parameters.extend(item_ids)
    rows = conn.execute(
        f"""
        SELECT i.* FROM items i
        WHERE {' AND '.join(clauses)}
        ORDER BY i.uid,i.item_index
        """,
        parameters,
    ).fetchall()
    if query_regex:
        pattern = re.compile(query_regex, flags=re.IGNORECASE)
        rows = [row for row in rows if pattern.search(str(row["found_by_query"] or ""))]
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["category"]), str(row["polarity"]))].append(row)
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


def transcript_for_item(project_root: Path, row: sqlite3.Row) -> list[dict[str, Any]]:
    """Return the aligned utterances in the demo interval plus one second of context."""
    path = project_root / "data" / "transcripts" / f"{row['uid']}.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    start = float(row["start_sec"] or 0.0) - 1.0
    end = float(row["end_sec"] or 0.0) + 1.0
    records = []
    for segment in payload.get("segments") or []:
        try:
            segment_start = float(segment["start"])
            segment_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if segment_end < start or segment_start > end:
            continue
        records.append(
            {
                "start": round(segment_start, 3),
                "end": round(segment_end, 3),
                "clip_start": round(segment_start - float(row["start_sec"] or 0.0), 3),
                "clip_end": round(segment_end - float(row["start_sec"] or 0.0), 3),
                "text": str(segment.get("text") or "").strip(),
            }
        )
    return records


def extract_frames(
    ffmpeg: str, ffprobe: str, clip: Path, frames_dir: Path, ordinal: int, max_frames: int
) -> tuple[float, list[dict[str, Any]]]:
    duration = probe_duration(ffprobe, clip)
    frame_count = min(max_frames, max(3, math.ceil(duration)))
    fractions = [(index + 0.5) / frame_count for index in range(frame_count)]
    frames = []
    for index, fraction in enumerate(fractions):
        timestamp = min(duration - 0.02, max(0.0, duration * fraction))
        name = f"{ordinal:04d}_f{index:02d}.jpg"
        target = frames_dir / name
        actual_timestamp = None
        errors = []
        for backoff in (0.0, 0.5, 1.0, 2.0, 4.0):
            candidate_timestamp = max(0.0, timestamp - backoff)
            target.unlink(missing_ok=True)
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{candidate_timestamp:.3f}",
                    "-i",
                    str(clip),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=768:-2",
                    "-strict",
                    "unofficial",
                    "-threads",
                    "1",
                    "-q:v",
                    "3",
                    "-y",
                    str(target),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and target.is_file() and target.stat().st_size > 0:
                actual_timestamp = candidate_timestamp
                break
            errors.append(result.stderr.strip()[-500:])
        if actual_timestamp is None:
            raise RuntimeError(f"failed to extract {clip} near {timestamp:.3f}s: {errors[-1]}")
        frames.append(
            {
                "frame_index": index,
                "timestamp": round(actual_timestamp, 3),
                "requested_timestamp": round(timestamp, 3),
                "path": f"frames/{name}",
            }
        )
    return duration, frames


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/visual_audit/audit.db"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--pillar", choices=("instructional", "witnessed"), default="instructional")
    parser.add_argument("--rubric", default="instructional_v3")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--seed", default="instructional-full-audit-v1")
    parser.add_argument("--polarity", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--query-regex", default="")
    parser.add_argument("--query-source", action="append", default=[])
    parser.add_argument("--uid", action="append", default=[])
    parser.add_argument("--item-id", action="append", default=[])
    parser.add_argument(
        "--item-id-manifest",
        type=Path,
        default=None,
        help="select the exact item_ids listed in an earlier audit manifest",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    if args.item_id_manifest is not None:
        prior = json.loads(args.item_id_manifest.read_text())
        args.item_id.extend(str(item["item_id"]) for item in prior.get("items") or [])

    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir()
    conn = connect(args.db)
    selected = select_balanced(
        conn,
        args.pillar,
        args.count,
        args.rubric,
        args.model,
        args.seed,
        args.polarity,
        args.category,
        args.query_regex,
        args.query_source,
        args.uid,
        args.item_id,
    )
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "instr" if args.pillar == "instructional" else "wit"
    batch_id = f"{prefix}_{stamp}_{hashlib.sha1(args.seed.encode()).hexdigest()[:8]}"
    records = []
    project_root = args.project_root.resolve()
    manifest_path = args.out / "manifest.json"
    conn.execute(
        """
        INSERT INTO batches(
            batch_id,pillar,rubric_version,intended_model,selection_strategy,
            seed,manifest_path,created_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            batch_id,
            args.pillar,
            args.rubric,
            args.model,
            "balanced_category_polarity_unjudged",
            args.seed,
            str(manifest_path.resolve()),
            time.time(),
        ),
    )
    for ordinal, row in enumerate(selected):
        clip = project_root / row["clip_path"]
        try:
            duration, frames = extract_frames(
                args.ffmpeg, args.ffprobe, clip, frames_dir, ordinal, args.max_frames
            )
            error = None
        except Exception as exc:
            duration, frames, error = None, [], str(exc)
        frame_hash = sha256_json(frames)
        record = {key: row[key] for key in row.keys()}
        record.update(
            {
                "ordinal": ordinal,
                "duration": duration,
                "frames": frames,
                "frame_manifest_sha256": frame_hash,
                "frame_error": error,
                "aligned_transcript": transcript_for_item(project_root, row),
            }
        )
        records.append(record)
        conn.execute(
            """
            INSERT INTO batch_items(batch_id,item_id,ordinal,frame_count,frame_manifest_sha256)
            VALUES (?,?,?,?,?)
            """,
            (batch_id, row["item_id"], ordinal, len(frames), frame_hash),
        )
    manifest = {
        "batch_id": batch_id,
        "pillar": args.pillar,
        "rubric_version": args.rubric,
        "intended_model": args.model,
        "selection_strategy": "balanced_category_polarity_unjudged",
        "prior_judgment_policy": "exclude_any_source_judgment_or_prior_batch_assignment",
        "filters": {
            "polarities": args.polarity,
            "categories": args.category,
            "query_regex": args.query_regex,
            "uids": args.uid,
        },
        "seed": args.seed,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "items": records,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    conn.commit()
    conn.close()
    print(json.dumps({"batch_id": batch_id, "items": len(records), "frames": sum(len(r["frames"]) for r in records)}))


if __name__ == "__main__":
    main()
