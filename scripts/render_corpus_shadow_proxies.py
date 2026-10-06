#!/usr/bin/env python3
"""Resumably render bounded video proxies for a frozen corpus manifest.

Successful render records are append-only. Failed attempts are retained and
retried on a later invocation. Once every manifest item has a successful
record, a complete VLM-ready JSONL manifest is materialized atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def successful_by_item(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        row["item_id"]: row
        for row in records
        if row.get("error") is None and row.get("proxy_clip")
    }


def latest_by_item(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["item_id"]: row for row in records if row.get("item_id")}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def proxy_name(item_id: str) -> str:
    digest = hashlib.sha256(item_id.encode()).hexdigest()[:12]
    safe = item_id.replace(":", "__").replace("/", "_")
    return f"{safe[:100]}__{digest}.mp4"


def probe_duration(source: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def sampling_fps(duration: float, max_fps: float, max_frames: int) -> float:
    if duration <= 0:
        raise ValueError(f"duration must be positive: {duration}")
    return min(max_fps, max_frames / duration)


def render_one(
    row: dict[str, Any],
    out_dir: Path,
    ffmpeg: str,
    ffprobe: str,
    max_fps: float,
    max_frames: int,
    max_side: int,
) -> dict[str, Any]:
    source = Path(row["source_clip"])
    target = out_dir / proxy_name(row["item_id"])
    try:
        if not source.is_file():
            raise FileNotFoundError(source)
        duration = float(row.get("duration_hint") or 0)
        if duration <= 0:
            duration = probe_duration(source, ffprobe)
        media_start = row.get("media_start_sec")
        media_end = row.get("media_end_sec")
        try:
            media_start = (
                max(0.0, float(media_start))
                if media_start is not None
                else None
            )
            media_end = (
                max(0.0, float(media_end))
                if media_end is not None
                else None
            )
        except (TypeError, ValueError):
            media_start = None
            media_end = None
        if (
            media_start is not None
            and media_end is not None
            and media_end > media_start
        ):
            duration = media_end - media_start
        fps = sampling_fps(duration, max_fps, max_frames)
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
        ]
        if media_start is not None:
            command.extend(["-ss", f"{media_start:.3f}"])
        command.extend(["-i", str(source)])
        if media_start is not None and media_end is not None:
            command.extend(["-t", f"{duration:.3f}"])
        command.extend(
            [
                "-an",
                "-vf",
                (
                    f"fps={fps:.4f},"
                    f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease,"
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2"
                ),
                "-frames:v",
                str(max_frames),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "30",
                "-movflags",
                "+faststart",
                str(target),
            ]
        )
        subprocess.run(command, check=True)
        return {
            "item_id": row["item_id"],
            "ordinal": row["ordinal"],
            "proxy_clip": str(target.resolve()),
            "proxy_size": target.stat().st_size,
            "proxy_sha256": file_sha256(target),
            "sampling_fps": fps,
            "duration": duration,
            "media_start_sec": media_start,
            "media_end_sec": media_end,
            "error": None,
        }
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {
            "item_id": row["item_id"],
            "ordinal": row["ordinal"],
            "proxy_clip": None,
            "proxy_size": None,
            "proxy_sha256": None,
            "sampling_fps": None,
            "duration": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_complete_manifest(
    source_rows: list[dict[str, Any]],
    successes: dict[str, dict[str, Any]],
    target: Path,
) -> None:
    lines = []
    for row in source_rows:
        merged = dict(row)
        merged.update(successes[row["item_id"]])
        lines.append(json.dumps(merged, sort_keys=True) + "\n")
    content = "".join(lines)
    if target.exists():
        if target.read_text() != content:
            raise ValueError(f"existing complete manifest differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--complete-manifest", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-fps", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--materialize-with-failures",
        action="store_true",
        help=(
            "Materialize an exact manifest after every item has at least one "
            "attempt, retaining failed rows for explicit unreadable-media routing."
        ),
    )
    args = parser.parse_args()

    source_rows = load_jsonl(args.manifest)
    existing_records = load_jsonl(args.records)
    completed = successful_by_item(existing_records)
    latest = latest_by_item(existing_records)
    pending = [row for row in source_rows if row["item_id"] not in completed]
    if args.materialize_with_failures:
        pending = [row for row in pending if row["item_id"] not in latest]
    if args.limit is not None:
        pending = pending[: args.limit]
    args.records.parent.mkdir(parents=True, exist_ok=True)

    with args.records.open("a") as handle, ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = {
            pool.submit(
                render_one,
                row,
                args.out_dir,
                args.ffmpeg,
                args.ffprobe,
                args.max_fps,
                args.max_frames,
                args.max_side,
            ): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            if index % 100 == 0 or result["error"]:
                print(
                    f"{index}/{len(futures)} {result['item_id']} "
                    f"{'ok' if result['error'] is None else result['error']}",
                    flush=True,
                )

    all_records = load_jsonl(args.records)
    completed = successful_by_item(all_records)
    latest = latest_by_item(all_records)
    missing = [row["item_id"] for row in source_rows if row["item_id"] not in completed]
    attempted_missing = [
        row["item_id"] for row in source_rows if row["item_id"] not in latest
    ]
    if not missing:
        write_complete_manifest(source_rows, completed, args.complete_manifest)
    elif args.materialize_with_failures and not attempted_missing:
        write_complete_manifest(source_rows, latest, args.complete_manifest)
    print(
        json.dumps(
            {
                "manifest_items": len(source_rows),
                "successful": len(completed),
                "remaining": len(missing),
                "failed": len(latest) - len(completed),
                "complete_manifest_written": (
                    not missing
                    or (args.materialize_with_failures and not attempted_missing)
                ),
                "corpus_mutated": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
