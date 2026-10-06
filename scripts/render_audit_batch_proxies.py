#!/usr/bin/env python3
"""Render small, append-only video proxies for a frozen audit batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


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


def sampling_fps(duration: float, max_fps: float, max_frames: int) -> float:
    if duration <= 0:
        raise ValueError(f"duration must be positive: {duration}")
    if max_fps <= 0 or max_frames <= 0:
        raise ValueError("max_fps and max_frames must be positive")
    return min(max_fps, max_frames / duration)


def render_proxy(
    source: Path,
    target: Path,
    ffmpeg: str,
    fps: float,
    max_side: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-an",
            "-vf",
            # Keep a compact rational time base. Long decimal frame rates can
            # make some MP4 muxers fail with an unsupported timestamp scale.
            (
                f"fps={fps:.2f},"
                f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease,"
                "scale=trunc(iw/2)*2:trunc(ih/2)*2"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
    )


def render_one(
    item: dict[str, Any],
    root: Path,
    out_dir: Path,
    ffmpeg: str,
    max_fps: float,
    max_frames: int,
    max_side: int,
) -> dict[str, Any]:
    source = root / item["clip_path"]
    if not source.is_file():
        raise FileNotFoundError(source)
    target = out_dir / proxy_name(item["item_id"])
    fps = sampling_fps(float(item["duration"]), max_fps, max_frames)
    render_proxy(source, target, ffmpeg, fps, max_side)
    return {
        "ordinal": int(item["ordinal"]),
        "item_id": item["item_id"],
        "uid": item["uid"],
        "source_clip_path": str(source),
        "source_size": source.stat().st_size,
        "proxy_path": str(target),
        "proxy_size": target.stat().st_size,
        "proxy_sha256": file_sha256(target),
        "sampling_fps": fps,
        "max_frames": max_frames,
        "max_side": max_side,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-fps", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--max-side", type=int, default=512)
    args = parser.parse_args()
    if args.out_manifest.exists():
        raise SystemExit(f"refusing to overwrite {args.out_manifest}")
    items = json.loads(args.manifest.read_text()).get("items") or []
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                render_one,
                item,
                args.root,
                args.out_dir,
                args.ffmpeg,
                args.max_fps,
                args.max_frames,
                args.max_side,
            ): item
            for item in items
        }
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            print(
                f"{index}/{len(items)} {record['item_id']} "
                f"{record['proxy_size']} bytes",
                flush=True,
            )
    records.sort(key=lambda row: row["ordinal"])
    if [row["ordinal"] for row in records] != list(range(len(items))):
        raise ValueError("rendered proxy ordinals are not complete and contiguous")
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "audit_batch_video_proxies",
                "source_manifest": str(args.manifest),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "proxy_bytes": sum(row["proxy_size"] for row in records),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
