#!/usr/bin/env python3
"""Run one conservative, resumable worker from a frozen YouTube manifest."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from backfill_commentary_videos import (
    VIDEO_SUFFIXES,
    append_ledger,
    download_one,
    utc_now,
)


AUTH_MARKERS = (
    "sign in to confirm you’re not a bot",
    "sign in to confirm you're not a bot",
    "account authentication is required",
    "login_required",
    "provided youtube account cookies are no longer valid",
)


def is_auth_failure(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in AUTH_MARKERS)


def load_records(manifest: Path, worker_id: str) -> list[dict]:
    records = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("worker") == worker_id:
            records.append(row)
    return records


def present_uids(stable_dir: Path) -> set[str]:
    return {
        path.stem for path in stable_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        and path.stat().st_size > 0
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--worker-id", choices=("sk1", "sk2", "sk3"), required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--cookies-file", type=Path, required=True)
    parser.add_argument("--js-runtime", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--sleep-seconds", type=float, default=10.0)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--max-duration", type=int, default=1800)
    parser.add_argument("--rate-limit", default="8M")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-free-bytes", type=int, default=0)
    parser.add_argument("--max-new-bytes", type=int, default=0)
    parser.add_argument("--max-consecutive-auth-failures", type=int, default=2)
    args = parser.parse_args()

    for path, label in ((args.manifest, "manifest"), (args.cookies_file, "cookies")):
        if not path.is_file():
            raise SystemExit(f"{label} file not found: {path}")
    if args.sleep_seconds < 0 or args.limit < 0:
        raise SystemExit("sleep-seconds and limit must be non-negative")

    # yt-dlp discovers ffmpeg beside this isolated Python environment.
    env_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")

    root = args.storage_root.resolve()
    stable_dir = root / "data" / "discussion_video"
    stage_dir = stable_dir / ".stage"
    ledger = stable_dir / f"youtube_backfill_{args.worker_id}.jsonl"
    stable_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(exist_ok=True)

    lock_handle = (stable_dir / f"youtube_backfill_{args.worker_id}.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"YouTube worker {args.worker_id} is already running") from exc

    records = load_records(args.manifest, args.worker_id)
    present = present_uids(stable_dir)
    downloaded = failed = skipped = attempted = new_bytes = 0
    consecutive_auth_failures = 0
    stop_reason = None

    for record in records:
        uid = record["uid"]
        if uid in present:
            skipped += 1
            continue
        if args.limit and downloaded >= args.limit:
            stop_reason = "limit"
            break
        if args.max_new_bytes and new_bytes >= args.max_new_bytes:
            stop_reason = "max_new_bytes"
            break
        free_bytes = shutil.disk_usage(stable_dir).free
        if args.min_free_bytes and free_bytes < args.min_free_bytes:
            stop_reason = "min_free_bytes"
            break

        attempted += 1
        started = time.monotonic()
        try:
            details = download_one(
                record, stage_dir, stable_dir, args.ffprobe,
                args.max_height, args.max_duration, args.rate_limit,
                args.cookies_file, None, args.js_runtime,
            )
            row = {
                "at": utc_now(), "status": "downloaded", "worker": args.worker_id,
                "uid": uid, "source": "youtube", "url": record.get("url"),
                "bucket": record.get("bucket"),
                "elapsed_sec": round(time.monotonic() - started, 3), **details,
            }
            append_ledger(ledger, row)
            print(json.dumps(row, sort_keys=True), flush=True)
            downloaded += 1
            new_bytes += int(details["bytes"])
            present.add(uid)
            consecutive_auth_failures = 0
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            message = str(exc)[-1500:]
            auth_failure = is_auth_failure(message)
            consecutive_auth_failures = (
                consecutive_auth_failures + 1 if auth_failure else 0
            )
            row = {
                "at": utc_now(), "status": "failed", "worker": args.worker_id,
                "uid": uid, "source": "youtube", "url": record.get("url"),
                "bucket": record.get("bucket"), "auth_failure": auth_failure,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "error": message,
            }
            append_ledger(ledger, row)
            print(json.dumps(row, sort_keys=True), flush=True)
            failed += 1
            if consecutive_auth_failures >= args.max_consecutive_auth_failures:
                stop_reason = "authentication_circuit_breaker"
                break
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

    summary = {
        "at": utc_now(), "status": "worker_summary", "worker": args.worker_id,
        "records": len(records), "attempted": attempted, "downloaded": downloaded,
        "failed": failed, "already_present": skipped, "new_bytes": new_bytes,
        "stop_reason": stop_reason or "manifest_exhausted",
        "free_bytes": shutil.disk_usage(stable_dir).free,
    }
    append_ledger(ledger, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
