#!/usr/bin/env python3
"""Build a small, stratified visual-QA bundle from the three corpus pillars.

Run this on the scraper host, where ``data/`` contains the live corpus.  The
bundle is intentionally frame-only: it is cheap to copy off sk3 and is enough
for a first-pass VLM audit.  Three temporal positions are sampled from every
selected clip to avoid treating a single midpoint as the whole clip.

Examples
--------
    python scripts/visual_audit_sample.py --out /tmp/norm-audit \
        --instructional 60 --witnessed 30 --commentary 30

    # Hourly audit: prefer items written in the last 90 minutes.
    python scripts/visual_audit_sample.py --out /tmp/norm-audit \
        --since-hours 1.5 --instructional 24 --witnessed 16 --commentary 16
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sqlite3
import subprocess
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def recent_enough(path: Path, cutoff: float | None) -> bool:
    return cutoff is None or path.stat().st_mtime >= cutoff


def prior_audit_source_uids(db_path: Path) -> set[tuple[str, str]]:
    """Return source videos already judged or assigned to any audit batch."""
    if not db_path.is_file():
        return set()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        queries = []
        if {"items", "batch_items"}.issubset(tables):
            queries.append(
                "SELECT DISTINCT i.pillar,i.uid FROM items i "
                "JOIN batch_items b ON b.item_id=i.item_id"
            )
        for table in (
            "judgments",
            "witnessed_judgments",
            "commentary_judgments",
            "instructional_repair_judgments",
        ):
            if {"items", table}.issubset(tables):
                queries.append(
                    f"SELECT DISTINCT i.pillar,i.uid FROM items i "
                    f"JOIN {table} j ON j.item_id=i.item_id"
                )
        rows = [] if not queries else conn.execute(" UNION ".join(queries)).fetchall()
        conn.close()
    except sqlite3.Error:
        return set()
    return {(str(pillar), str(uid)) for pillar, uid in rows}


def exclude_prior_sources(
    rows: Iterable[dict[str, Any]], prior_sources: set[tuple[str, str]]
) -> tuple[list[dict[str, Any]], int]:
    kept = []
    excluded = 0
    for row in rows:
        key = (str(row.get("pillar") or ""), str(row.get("uid") or ""))
        if key in prior_sources:
            excluded += 1
        else:
            kept.append(row)
    return kept, excluded


def load_uid_exclusions(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                excluded.add(value)
    return excluded


def exclude_uids(
    rows: Iterable[dict[str, Any]], excluded_uids: set[str]
) -> tuple[list[dict[str, Any]], int]:
    kept = []
    excluded = 0
    for row in rows:
        if str(row.get("uid") or "") in excluded_uids:
            excluded += 1
        else:
            kept.append(row)
    return kept, excluded


def balanced_sample(
    rows: Iterable[dict[str, Any]],
    n: int,
    strata: tuple[str, ...],
    rng: random.Random,
    unique_by: str | None = None,
) -> list[dict[str, Any]]:
    """Round-robin across observed strata, with randomized order within each."""
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in strata)].append(row)
    queues: list[deque[dict[str, Any]]] = []
    for key in sorted(buckets, key=lambda x: tuple(str(v) for v in x)):
        rng.shuffle(buckets[key])
        queues.append(deque(buckets[key]))
    rng.shuffle(queues)

    chosen: list[dict[str, Any]] = []
    seen: set[Any] = set()
    while queues and len(chosen) < n:
        next_round: list[deque[dict[str, Any]]] = []
        for queue in queues:
            if len(chosen) >= n:
                break
            if queue:
                candidate = queue.popleft()
                identity = candidate.get(unique_by) if unique_by else None
                if unique_by is None or identity not in seen:
                    chosen.append(candidate)
                    seen.add(identity)
            if queue:
                next_round.append(queue)
        queues = next_round
    return chosen


def instructional_rows(root: Path, cutoff: float | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta_path in root.glob("*/metadata.json"):
        if not recent_enough(meta_path, cutoff):
            continue
        meta = load_json(meta_path)
        if not meta:
            continue
        uid = meta.get("video_id") or meta_path.parent.name
        candidates = []
        for idx, demo in enumerate(meta.get("demos") or []):
            clip_name = demo.get("clip")
            if not clip_name:
                continue
            clip_path = meta_path.parent / str(clip_name)
            if not clip_path.is_file():
                continue
            candidates.append(
                {
                    "pillar": "instructional",
                    "uid": uid,
                    "title": meta.get("title"),
                    "category": meta.get("category") or (meta.get("provenance") or {}).get("category"),
                    "query_source": (meta.get("provenance") or {}).get("query_source"),
                    "genre": meta.get("genre"),
                    "polarity": demo.get("polarity"),
                    "norm": demo.get("norm"),
                    "start_quote": demo.get("start_quote"),
                    "end_quote": demo.get("end_quote"),
                    "explanation": demo.get("explanation"),
                    "clip_index": idx,
                    "clip_path": str(clip_path),
                    "metadata_mtime": meta_path.stat().st_mtime,
                }
            )
        rows.extend(candidates)
    return rows


def database_titles(data_root: Path) -> dict[str, str]:
    db_path = data_root / "state.db"
    if not db_path.is_file():
        return {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT video_id, title FROM seen_videos WHERE title IS NOT NULL"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return {}
    return {str(uid): str(title) for uid, title in rows}


def witnessed_rows(root: Path, cutoff: float | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta_path in root.glob("*/metadata.json"):
        if not recent_enough(meta_path, cutoff):
            continue
        meta = load_json(meta_path)
        if not meta:
            continue
        uid = meta.get("video_id") or meta_path.parent.name
        provenance = meta.get("provenance") or {}
        scene = provenance.get("scene") or {}
        reactions = meta.get("reactions") or []
        by_clip: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for reaction in reactions:
            try:
                by_clip[int(reaction.get("clip_idx", 0))].append(reaction)
            except (TypeError, ValueError):
                by_clip[0].append(reaction)
        clips = sorted(meta_path.parent.glob("clip_*.mp4"))
        if not clips:
            clips = sorted(meta_path.parent.glob("*.mp4"))
        if not clips:
            continue
        # One reaction window per source video gives an independent-video ratio.
        clip_path = clips[0]
        try:
            clip_idx = int(clip_path.stem.rsplit("_", 1)[-1])
        except ValueError:
            clip_idx = 0
        clip_reactions = by_clip.get(clip_idx) or reactions[:1]
        rows.append(
            {
                "pillar": "witnessed",
                "uid": uid,
                "title": meta.get("title"),
                "category": provenance.get("category"),
                "query_source": provenance.get("query_source"),
                "query": provenance.get("found_by_query"),
                "agent": meta.get("agent"),
                "scene_type": scene.get("scene_type"),
                "n_people": scene.get("n_people"),
                "violator_role": scene.get("violator_role"),
                "reactor_role": scene.get("reactor_role"),
                "severity": scene.get("severity"),
                "reaction_strength": scene.get("reaction_strength"),
                "norm": (clip_reactions[0].get("norm") if clip_reactions else None),
                "reaction": (clip_reactions[0].get("phrase") if clip_reactions else None),
                "reaction_context": (clip_reactions[0].get("context") if clip_reactions else None),
                "clip_index": clip_idx,
                "clip_path": str(clip_path),
                "metadata_mtime": meta_path.stat().st_mtime,
            }
        )
    return rows


def retained_video(video_root: Path, uid: str) -> Path | None:
    matches = [
        path
        for path in video_root.glob(f"{uid}.*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
        and path.stat().st_size > 0
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple retained commentary videos for {uid}: {matches}")
    return matches[0] if matches else None


def commentary_rows(
    root: Path,
    cutoff: float | None,
    video_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return commentary sources with retained video, not text-only records."""
    rows: list[dict[str, Any]] = []
    video_root = video_root or root.parent / "discussion_video"
    meta_paths = list(root.glob("*.json"))
    if cutoff is not None:
        # An hourly audit only needs recently written metadata or an older
        # record whose source video was newly retained. Filter by filesystem
        # timestamps before parsing tens of thousands of historical JSON files.
        recent_video_uids = {
            path.stem
            for path in video_root.iterdir()
            if path.is_file()
            and path.suffix.lower() in VIDEO_SUFFIXES
            and path.stat().st_size > 0
            and recent_enough(path, cutoff)
        }
        meta_paths = [
            path
            for path in meta_paths
            if recent_enough(path, cutoff) or path.stem in recent_video_uids
        ]
    for meta_path in meta_paths:
        meta = load_json(meta_path)
        if not meta:
            continue
        uid = str(meta.get("video_id") or meta_path.stem)
        video = retained_video(video_root, uid)
        if video is None:
            continue
        # A newly retained copy of an older discussion record is fresh visual
        # material and must be eligible for an hourly audit.
        provenance = meta.get("provenance") or {}
        scene = provenance.get("scene") or {}
        statements = meta.get("statements") or []
        rows.append(
            {
                "pillar": "commentary",
                "uid": uid,
                "title": meta.get("title"),
                "category": meta.get("category") or provenance.get("category"),
                "query_source": provenance.get("query_source"),
                "query": provenance.get("found_by_query"),
                "agent": meta.get("agent"),
                "scene_type": scene.get("scene_type"),
                "n_people": scene.get("n_people"),
                "norm": (statements[0].get("norm") if statements else None),
                "statement": (statements[0].get("quote") if statements else None),
                "signal": (statements[0].get("signal") if statements else None),
                "n_statements": meta.get("n_statements", len(statements)),
                "clip_path": str(video),
                "retained_video_mtime": video.stat().st_mtime,
                "metadata_mtime": meta_path.stat().st_mtime,
            }
        )
    return rows


def probe_duration(ffprobe: str, clip: Path) -> float | None:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(clip),
        ],
        capture_output=True,
        text=True,
    )
    try:
        value = float(result.stdout.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def extract_frames(
    row: dict[str, Any], audit_index: int, out_dir: Path, positions: list[float], ffmpeg: str, ffprobe: str
) -> list[str]:
    clip = Path(row["clip_path"])
    duration = probe_duration(ffprobe, clip)
    row["duration"] = duration
    if duration is None:
        row["frame_error"] = "could not probe duration"
        return []
    frame_names: list[str] = []
    for frame_index, fraction in enumerate(positions):
        timestamp = min(max(duration * fraction, 0.0), max(duration - 0.05, 0.0))
        name = f"{audit_index:03d}_{row['pillar']}_{frame_index}_{row['uid']}.jpg"
        target = out_dir / name
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(clip),
                # Output-side seeking is slower but frame-accurate. Input-side
                # seeking can silently return a later keyframe and previously
                # caused audit sheets to omit the beginning of target clips.
                "-ss",
                f"{timestamp:.3f}",
                "-frames:v",
                "1",
                "-vf",
                "scale=640:-2",
                "-q:v",
                "3",
                "-y",
                str(target),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and target.is_file():
            frame_names.append(name)
        else:
            row.setdefault("frame_errors", []).append(result.stderr[-300:])
    return frame_names


def requested_rows(count: int, loader):
    """Avoid an expensive corpus traversal when a pillar count is zero."""
    return loader() if count else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--audit-db",
        type=Path,
        default=None,
        help="audit ledger; defaults to DATA_ROOT/visual_audit/audit.db",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instructional", type=int, default=60)
    parser.add_argument("--witnessed", type=int, default=30)
    parser.add_argument("--commentary", type=int, default=30)
    parser.add_argument("--since-hours", type=float, default=0.0)
    parser.add_argument("--seed", default="visual-audit-v1")
    parser.add_argument(
        "--exclude-uids",
        type=Path,
        action="append",
        default=[],
        help="Plain-text source UID lists to exclude in addition to the audit ledger.",
    )
    parser.add_argument("--positions", default="0.15,0.50,0.85")
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--ffprobe", default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=False)
    frames_dir = args.out / "frames"
    frames_dir.mkdir()
    rng = random.Random(args.seed)
    cutoff = None
    if args.since_hours > 0:
        cutoff = datetime.now(tz=timezone.utc).timestamp() - args.since_hours * 3600

    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    ffprobe = args.ffprobe or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe must be available (or pass their full paths)")

    # Hourly single-pillar audits should not traverse the other two large
    # corpus trees. A requested count of zero is an explicit skip, not merely
    # an empty sample after a full scan.
    raw_instructional_pool = requested_rows(
        args.instructional,
        lambda: instructional_rows(args.data_root / "instructional", cutoff),
    )
    raw_witnessed_pool = requested_rows(
        args.witnessed,
        lambda: witnessed_rows(args.data_root / "hits", cutoff),
    )
    raw_commentary_pool = requested_rows(
        args.commentary,
        lambda: commentary_rows(
            args.data_root / "discussion",
            cutoff,
            args.data_root / "discussion_video",
        ),
    )
    audit_db = args.audit_db or args.data_root / "visual_audit" / "audit.db"
    prior_sources = prior_audit_source_uids(audit_db)
    instructional_pool, instructional_excluded = exclude_prior_sources(
        raw_instructional_pool, prior_sources
    )
    witnessed_pool, witnessed_excluded = exclude_prior_sources(
        raw_witnessed_pool, prior_sources
    )
    commentary_pool, commentary_excluded = exclude_prior_sources(
        raw_commentary_pool, prior_sources
    )
    explicit_exclusions = load_uid_exclusions(args.exclude_uids)
    instructional_pool, instructional_explicit_excluded = exclude_uids(
        instructional_pool, explicit_exclusions
    )
    witnessed_pool, witnessed_explicit_excluded = exclude_uids(
        witnessed_pool, explicit_exclusions
    )
    commentary_pool, commentary_explicit_excluded = exclude_uids(
        commentary_pool, explicit_exclusions
    )
    instructional = balanced_sample(
        instructional_pool,
        args.instructional,
        ("category", "polarity"),
        rng,
        unique_by="uid",
    )
    witnessed = balanced_sample(
        witnessed_pool, args.witnessed, ("query_source", "category", "scene_type"), rng
    )
    commentary = balanced_sample(
        commentary_pool, args.commentary, ("query_source", "category"), rng
    )
    titles = database_titles(args.data_root)
    visual_rows = instructional + witnessed + commentary
    positions = [float(value) for value in args.positions.split(",")]
    for index, row in enumerate(visual_rows):
        row["audit_index"] = index
        row["title"] = row.get("title") or titles.get(str(row.get("uid")))
        row["frames"] = extract_frames(row, index, frames_dir, positions, ffmpeg, ffprobe)
    for index, row in enumerate(commentary):
        row["commentary_index"] = index
        row["title"] = row.get("title") or titles.get(str(row.get("uid")))

    manifest = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "seed": args.seed,
        "since_hours": args.since_hours,
        "frame_positions": positions,
        "prior_judgment_policy": "exclude_any_source_with_prior_judgment_or_batch_assignment",
        "explicit_exclusion_files": [str(path.resolve()) for path in args.exclude_uids],
        "explicit_excluded_source_uids": len(explicit_exclusions),
        "commentary_policy": "retained_video_required_and_temporally_sampled",
        "raw_pool_counts": {
            "instructional": len(raw_instructional_pool),
            "witnessed": len(raw_witnessed_pool),
            "commentary": len(raw_commentary_pool),
        },
        "excluded_prior_source_rows": {
            "instructional": instructional_excluded,
            "witnessed": witnessed_excluded,
            "commentary": commentary_excluded,
        },
        "excluded_explicit_source_rows_after_ledger": {
            "instructional": instructional_explicit_excluded,
            "witnessed": witnessed_explicit_excluded,
            "commentary": commentary_explicit_excluded,
        },
        "pool_counts": {
            "instructional": len(instructional_pool),
            "witnessed": len(witnessed_pool),
            "commentary": len(commentary_pool),
        },
        "sample_counts": {
            "instructional": len(instructional),
            "witnessed": len(witnessed),
            "commentary": len(commentary),
        },
        "visual_samples": visual_rows,
        "commentary_samples": commentary,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    with (args.out / "visual_review.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "audit_index",
                "pillar",
                "visual_demo_present",
                "medium",
                "norm_relevant",
                "organic_scene",
                "weak_label_valid",
                "note",
            ]
        )
        for row in visual_rows:
            writer.writerow([row["audit_index"], row["pillar"], "", "", "", "", "", ""])
    with (args.out / "commentary_review.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "commentary_index",
                "uid",
                "normative_statement_valid",
                "concrete_social_behavior",
                "quote_supports_label",
                "note",
            ]
        )
        for row in commentary:
            writer.writerow([row["commentary_index"], row["uid"], "", "", "", ""])
    print(json.dumps(manifest["pool_counts"], sort_keys=True))
    print(json.dumps(manifest["sample_counts"], sort_keys=True))
    print(args.out)


if __name__ == "__main__":
    main()
