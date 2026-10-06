#!/usr/bin/env python3
"""Record an append-only collection-health snapshot and fresh review queue.

This monitor is deliberately observational.  It never restarts jobs, downloads
media, changes labels, or promotes audit candidates.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov"}
SOURCES = ("dailymotion", "rumble", "youtube", "reddit")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def process_count(pattern: str) -> int:
    result = subprocess.run(
        ["pgrep", "-fc", pattern], capture_output=True, text=True, check=False
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def source_inventory(discussion: Path, videos: Path) -> dict[str, dict[str, int]]:
    result = {}
    for source in SOURCES:
        records = sum(1 for _ in discussion.glob(f"{source}__*.json"))
        retained = {
            path.stem
            for path in videos.glob(f"{source}__*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_SUFFIXES
            and path.stat().st_size > 0
        }
        result[source] = {
            "records": records,
            "retained": len(retained),
            "missing": max(records - len(retained), 0),
        }
    return result


def ledger_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = row.get("uid")
        if uid:
            latest[str(uid)] = row
    statuses = Counter(row.get("status") for row in latest.values())
    return {
        "events": len(rows),
        "unique_downloaded": statuses["downloaded"],
        "latest_failed": statuses["failed"],
        "auth_failures_total": sum(bool(row.get("auth_failure")) for row in rows),
        "last_event_at": rows[-1].get("at") if rows else None,
        "last_event_status": rows[-1].get("status") if rows else None,
    }


def prior_queued_uids(snapshots: Path) -> set[str]:
    queued = set()
    for row in read_jsonl(snapshots):
        for item in row.get("fresh_review_queue") or []:
            if item.get("uid"):
                queued.add(str(item["uid"]))
    return queued


def fresh_review_queue(
    discussion: Path, videos: Path, snapshots: Path, count: int
) -> list[dict[str, Any]]:
    prior = prior_queued_uids(snapshots)
    candidates = sorted(
        (
            path
            for path in videos.iterdir()
            if path.is_file()
            and path.suffix.lower() in VIDEO_SUFFIXES
            and path.stat().st_size > 0
            and path.stem not in prior
        ),
        key=lambda path: (-path.stat().st_mtime_ns, path.name),
    )
    result = []
    for path in candidates[:count]:
        metadata = read_json(discussion / f"{path.stem}.json")
        result.append(
            {
                "uid": path.stem,
                "source": path.stem.split("__", 1)[0],
                "title": metadata.get("title"),
                "media_path": str(path),
                "media_bytes": path.stat().st_size,
                "media_mtime": path.stat().st_mtime,
                "label_status": "source_only_unlabeled",
            }
        )
    return result


def build_snapshot(root: Path, out_dir: Path, sample_size: int) -> dict[str, Any]:
    data = root / "data"
    discussion = data / "discussion"
    videos = data / "discussion_video"
    snapshots = out_dir / "snapshots.jsonl"
    inventory = source_inventory(discussion, videos)
    free_bytes = shutil.disk_usage(videos).free
    backfill_running = process_count("[b]ackfill_commentary_videos.py")
    crawl_running = process_count("[s]rc.search_loop")
    alerts = []
    if inventory["dailymotion"]["missing"] + inventory["rumble"]["missing"]:
        if backfill_running == 0:
            alerts.append("dailymotion_rumble_backfill_not_running_before_coverage_complete")
    if free_bytes < 10_000_000_000_000:
        alerts.append("discussion_video_filesystem_below_10TB_free")
    general = ledger_summary(videos / "backfill.jsonl")
    youtube = ledger_summary(videos / "youtube_backfill_sk3.jsonl")
    if general["auth_failures_total"] or youtube["auth_failures_total"]:
        alerts.append("authentication_failure_recorded")
    if crawl_running == 0:
        alerts.append("main_crawl_not_running")
    return {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "inventory": inventory,
        "processes": {
            "commentary_backfill": backfill_running,
            "main_crawl": crawl_running,
        },
        "ledgers": {"general": general, "youtube_sk3": youtube},
        "discussion_video_free_bytes": free_bytes,
        "alerts": alerts,
        "fresh_review_queue": fresh_review_queue(
            discussion, videos, snapshots, sample_size
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=4)
    args = parser.parse_args()
    if args.sample_size < 0:
        raise SystemExit("sample-size must be non-negative")
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = (out_dir / "monitor.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("hourly collection monitor is already running") from exc
    snapshot = build_snapshot(root, out_dir, args.sample_size)
    with (out_dir / "snapshots.jsonl").open("a") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n")
    temporary = out_dir / f"latest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, out_dir / "latest.json")
    print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
