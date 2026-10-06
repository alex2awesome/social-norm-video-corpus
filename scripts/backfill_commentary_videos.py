#!/usr/bin/env python3
"""Resumably retain/backfill full source videos for the commentary corpus."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_video(stable_dir: Path, uid: str) -> Path | None:
    matches = [
        path for path in stable_dir.glob(f"{uid}.*")
        if path.suffix.lower() in VIDEO_SUFFIXES and path.stat().st_size > 0
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple retained videos for {uid}: {matches}")
    return matches[0] if matches else None


def raw_video(raw_dir: Path, uid: str) -> Path | None:
    matches = [
        path for path in raw_dir.glob(f"{uid}.*")
        if path.suffix.lower() in VIDEO_SUFFIXES and path.stat().st_size > 0
    ]
    return max(matches, key=lambda path: path.stat().st_size) if matches else None


def iter_records(discussion_dir: Path) -> Iterable[dict]:
    records = []
    for path in discussion_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        uid = str(record.get("video_id") or path.stem)
        records.append({
            "uid": uid,
            "url": record.get("url"),
            "source": record.get("source"),
        })
    # Stable pseudo-random order avoids silently front-loading old IDs and makes
    # any prefix a defensible cross-platform canary.
    records.sort(key=lambda row: hashlib.sha256(row["uid"].encode()).hexdigest())
    return records


def append_ledger(ledger: Path, row: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def adopt_existing(raw: Path, target: Path) -> str:
    """Hard-link an existing raw source when possible, preserving the original."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(raw, target)
        return "hardlink"
    except OSError:
        tmp = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(raw, tmp)
        tmp.replace(target)
        return "copy"


def probe_media(ffprobe: str, path: Path) -> dict:
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-show_entries", "stream=codec_type", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-1000:])
    data = json.loads(proc.stdout)
    duration = float((data.get("format") or {}).get("duration") or 0)
    stream_types = [stream.get("codec_type") for stream in data.get("streams", [])]
    if duration <= 0 or "video" not in stream_types:
        raise RuntimeError(f"invalid media duration/streams: {duration}, {stream_types}")
    return {
        "duration": round(duration, 3),
        "video_streams": stream_types.count("video"),
        "audio_streams": stream_types.count("audio"),
    }


def download_one(
    record: dict,
    stage_dir: Path,
    stable_dir: Path,
    ffprobe: str,
    max_height: int,
    max_duration: int,
    rate_limit: str,
    cookies_file: Path | None = None,
    youtube_player_client: str | None = None,
    js_runtime: str | None = None,
) -> dict:
    uid = record["uid"]
    url = record.get("url")
    if not url:
        raise ValueError("missing source URL")
    for stale in stage_dir.glob(f"{uid}.*"):
        if stale.is_file():
            stale.unlink()
    output = stage_dir / f"{uid}.%(ext)s"
    command = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", str(Path(sys.executable).parent),
        "--no-playlist", "--no-progress", "--no-overwrites",
        "--retries", "5", "--fragment-retries", "5",
        "--socket-timeout", "30", "--sleep-requests", "1",
        "--limit-rate", rate_limit,
        "--match-filter", f"duration <? {max_duration} & !is_live",
        "-f", (
            f"bestvideo[height<={max_height}]+bestaudio/best/"
            f"best[height<={max_height}]/best"
        ),
        "--merge-output-format", "mp4", "--remux-video", "mp4",
        "--write-info-json", "-o", str(output), url,
    ]
    if cookies_file is not None:
        command[3:3] = ["--cookies", str(cookies_file)]
    if youtube_player_client:
        command[3:3] = [
            "--extractor-args", f"youtube:player_client={youtube_player_client}",
        ]
    if js_runtime:
        command[3:3] = ["--js-runtimes", js_runtime]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-1500:])
    candidates = [
        path for path in stage_dir.glob(f"{uid}.*")
        if path.suffix.lower() in VIDEO_SUFFIXES and path.stat().st_size > 0
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one downloaded video, got {candidates}")
    staged = candidates[0]
    media = probe_media(ffprobe, staged)
    target = stable_dir / f"{uid}{staged.suffix.lower()}"
    staged.replace(target)
    info = stage_dir / f"{uid}.info.json"
    if info.exists():
        info.replace(stable_dir / info.name)
    return {
        **media,
        "path": target.name,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum new downloads; 0 means all missing records")
    parser.add_argument("--max-attempts", type=int, default=0,
                        help="Maximum URL attempts; 0 means no attempt cap")
    parser.add_argument("--only-adopt", action="store_true")
    parser.add_argument("--source", action="append",
                        help="Restrict to a source platform; repeat for multiple")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--max-duration", type=int, default=1800)
    parser.add_argument("--rate-limit", default="8M")
    parser.add_argument("--cookies-file", type=Path,
                        help="Sanctioned Netscape cookies file for authenticated platforms")
    parser.add_argument("--youtube-player-client", default="android",
                        help="yt-dlp YouTube player client; empty disables the override")
    parser.add_argument("--js-runtime",
                        help="yt-dlp JavaScript runtime, e.g. deno:/absolute/path/to/deno")
    parser.add_argument("--ffprobe", default=str(Path(sys.executable).with_name("ffprobe")))
    args = parser.parse_args()
    if args.limit < 0 or args.max_attempts < 0 or args.sleep_seconds < 0:
        raise SystemExit("limit, max-attempts, and sleep-seconds must be non-negative")
    if args.cookies_file is not None:
        args.cookies_file = args.cookies_file.expanduser().resolve()
        if not args.cookies_file.is_file():
            raise SystemExit(f"cookies file does not exist: {args.cookies_file}")

    root = args.root.resolve()
    discussion_dir = root / "data" / "discussion"
    raw_dir = root / "data" / "raw_video"
    stable_dir = root / "data" / "discussion_video"
    stage_dir = stable_dir / ".stage"
    ledger = stable_dir / "backfill.jsonl"
    stable_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(exist_ok=True)

    lock_handle = (stable_dir / "backfill.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another commentary backfill is already running") from exc

    records = list(iter_records(discussion_dir))
    if args.source:
        allowed_sources = set(args.source)
        records = [row for row in records if row.get("source") in allowed_sources]
    present_uids = {
        path.stem for path in stable_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        and path.stat().st_size > 0
    }
    adopted = downloaded = failed = skipped = attempted = 0
    by_source: dict[str, dict[str, int]] = {}
    for record in records:
        uid = record["uid"]
        source = str(record.get("source") or "unknown")
        source_counts = by_source.setdefault(
            source, {"adopted": 0, "downloaded": 0, "failed": 0, "skipped": 0}
        )
        if uid in present_uids:
            skipped += 1
            source_counts["skipped"] += 1
            continue
        existing = raw_video(raw_dir, uid)
        if existing is not None:
            target = stable_dir / f"{uid}{existing.suffix.lower()}"
            method = adopt_existing(existing, target)
            details = probe_media(args.ffprobe, target)
            append_ledger(ledger, {
                "at": utc_now(), "status": "adopted", "uid": uid,
                "source": source, "path": target.name, "method": method,
                "bytes": target.stat().st_size, "sha256": sha256_file(target),
                **details,
            })
            adopted += 1
            present_uids.add(uid)
            source_counts["adopted"] += 1
            continue
        if (
            args.only_adopt
            or (args.limit and downloaded >= args.limit)
            or (args.max_attempts and attempted >= args.max_attempts)
        ):
            continue
        attempted += 1
        started = time.monotonic()
        try:
            details = download_one(
                record, stage_dir, stable_dir, args.ffprobe,
                args.max_height, args.max_duration, args.rate_limit,
                args.cookies_file, args.youtube_player_client or None,
                args.js_runtime,
            )
            append_ledger(ledger, {
                "at": utc_now(), "status": "downloaded", "uid": uid,
                "source": source, "url": record.get("url"),
                "elapsed_sec": round(time.monotonic() - started, 3), **details,
            })
            downloaded += 1
            present_uids.add(uid)
            source_counts["downloaded"] += 1
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            append_ledger(ledger, {
                "at": utc_now(), "status": "failed", "uid": uid,
                "source": source, "url": record.get("url"),
                "elapsed_sec": round(time.monotonic() - started, 3),
                "error": str(exc)[-1500:],
            })
            failed += 1
            source_counts["failed"] += 1
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

    print(json.dumps({
        "records": len(records), "adopted": adopted, "attempted": attempted,
        "downloaded": downloaded,
        "failed": failed, "already_present": skipped, "by_source": by_source,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
