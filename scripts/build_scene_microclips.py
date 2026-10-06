#!/usr/bin/env python3
"""Build non-destructive overlapping microclips for visual-scene follow-up.

The output is a derived benchmark manifest plus small proxy videos.  Source
videos and source metadata are never modified.  Existing microclips are reused,
which makes repeated calibration runs cheap and resumable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def video_timing(path: Path) -> tuple[float, float]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,start_time,duration:format=start_time,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return probe_video_seek_offset(payload), probe_duration(payload)


def duration(path: Path) -> float:
    """Backward-compatible duration helper for callers that do not seek."""
    return video_timing(path)[1]


def probe_video_start(payload: dict) -> float:
    """Return the input timestamp at which the first video stream begins."""
    for stream in payload.get("streams") or []:
        if stream.get("codec_type") != "video":
            continue
        value = stream.get("start_time")
        if value not in (None, "N/A"):
            return float(value)
    container = payload.get("format") or {}
    value = container.get("start_time")
    return 0.0 if value in (None, "N/A") else float(value)


def probe_video_seek_offset(payload: dict) -> float:
    """Return video start relative to the container's input-seek origin.

    ffmpeg's input-side ``-ss`` is relative to the container start.  A file
    whose video and container both start at timestamp 8 therefore needs a zero
    offset, while a file with audio/container at 0 and video at 8 needs 8.
    """
    container = payload.get("format") or {}
    value = container.get("start_time")
    container_start = 0.0 if value in (None, "N/A") else float(value)
    return max(0.0, probe_video_start(payload) - container_start)


def probe_duration(payload: dict) -> float:
    """Return playable video duration, not container end timestamp.

    Some extracted proxy MP4s preserve a positive stream start time.  In those
    files ``format.duration`` is the absolute end timestamp and overstates the
    number of decodable seconds.  Prefer the video stream duration and only
    fall back to subtracting the nonnegative container start time.
    """
    for stream in payload.get("streams") or []:
        if stream.get("codec_type") != "video":
            continue
        value = stream.get("duration")
        if value not in (None, "N/A"):
            parsed = float(value)
            if parsed > 0:
                return parsed
    container = payload.get("format") or {}
    value = container.get("duration")
    if value in (None, "N/A"):
        raise ValueError("ffprobe returned no usable video or container duration")
    start = container.get("start_time")
    playable = float(value) - max(0.0, float(start or 0.0))
    if playable <= 0:
        raise ValueError(f"nonpositive playable duration: {playable}")
    return playable


def windows(total: float, size: float, stride: float) -> list[tuple[float, float]]:
    if total <= size:
        return [(0.0, total)]
    starts: list[float] = []
    value = 0.0
    while value + size < total:
        starts.append(value)
        value += stride
    final_start = max(0.0, total - size)
    if not starts or abs(starts[-1] - final_start) > 0.25:
        starts.append(final_start)
    return [(start, min(total, start + size)) for start in starts]


def resolve_clip(row: dict, manifest: Path) -> Path:
    raw = Path(row.get("proxy_clip") or row["source_clip"])
    local = manifest.parent / "clips" / raw.name
    if local.exists():
        return local
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=6.0)
    parser.add_argument("--stride-sec", type=float, default=3.0)
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--pillar", action="append", default=[])
    parser.add_argument(
        "--max-source-sec",
        type=float,
        help="Only window the first N seconds of each selected source clip.",
    )
    parser.add_argument("--item-id", action="append", default=[])
    parser.add_argument(
        "--selection-jsonl",
        type=Path,
        help="Optional JSONL whose item_id values select parent clips.",
    )
    args = parser.parse_args()

    selected = set(args.item_id)
    if args.selection_jsonl:
        selected.update(row["item_id"] for row in load_jsonl(args.selection_jsonl))
    if not selected:
        raise SystemExit("select at least one --item-id or --selection-jsonl")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = args.out_dir / "clips"
    clip_dir.mkdir(exist_ok=True)
    output_manifest = args.out_dir / "manifest.jsonl"

    parent_rows = {
        row["item_id"]: row
        for row in load_jsonl(args.manifest)
        if row["item_id"] in selected
        and (not args.pillar or row["pillar"] in args.pillar)
    }
    missing = (
        selected - parent_rows.keys()
        if not args.pillar
        else set()
    )
    if missing:
        raise SystemExit(f"item IDs absent from manifest: {sorted(missing)}")

    output: list[dict] = []
    for item_id in sorted(parent_rows):
        parent = parent_rows[item_id]
        source = resolve_clip(parent, args.manifest)
        video_start, total = video_timing(source)
        if args.max_source_sec is not None:
            total = min(total, args.max_source_sec)
        spans = windows(total, args.window_sec, args.stride_sec)
        if len(spans) > args.max_windows:
            # Evenly retain temporal coverage without silently sampling only the
            # beginning of an unusually long clip.
            indexes = {
                round(i * (len(spans) - 1) / (args.max_windows - 1))
                for i in range(args.max_windows)
            }
            spans = [span for i, span in enumerate(spans) if i in indexes]
        safe_parent = (
            item_id.replace(":", "__").replace("/", "_").replace("\\", "_")
        )
        for index, (start, end) in enumerate(spans):
            target = clip_dir / f"{safe_parent}__w{index:03d}.mp4"
            if not target.exists():
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        f"{video_start + start:.3f}",
                        "-i",
                        str(source),
                        "-t",
                        f"{end - start:.3f}",
                        "-an",
                        "-vf",
                        "fps=4,scale=640:-2",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "25",
                        "-movflags",
                        "+faststart",
                        str(target),
                    ],
                    check=True,
                )
            row = dict(parent)
            row.update(
                {
                    "item_id": f"{item_id}:window:{index}",
                    "parent_item_id": item_id,
                    "window_index": index,
                    "window_start_sec": start,
                    "window_end_sec": end,
                    "proxy_clip": str(target.resolve()),
                    "source_proxy_clip": str(source.resolve()),
                }
            )
            output.append(row)

    with output_manifest.open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "parents": len(parent_rows),
                "microclips": len(output),
                "manifest": str(output_manifest),
                "clip_dir": str(clip_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
