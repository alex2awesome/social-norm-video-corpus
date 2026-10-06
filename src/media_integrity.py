"""Read-only media timing and playable-video integrity checks."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_media_timing(payload: dict[str, Any]) -> dict[str, Any]:
    streams = payload.get("streams") or []
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    audio = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    format_data = payload.get("format") or {}
    format_start = _number(format_data.get("start_time"))
    format_duration = _number(format_data.get("duration"))

    timed_streams = []
    for stream in streams:
        start = _number(stream.get("start_time"))
        duration = _number(stream.get("duration"))
        if start is not None and duration is not None and duration >= 0:
            timed_streams.append((start, start + duration))
    if timed_streams:
        timeline_start = min(start for start, _end in timed_streams)
        timeline_end = max(end for _start, end in timed_streams)
    else:
        timeline_start = format_start or 0.0
        timeline_end = (
            timeline_start + format_duration
            if format_duration is not None
            else timeline_start
        )
    timeline_duration = max(0.0, timeline_end - timeline_start)

    video_start = _number(video.get("start_time")) if video else None
    video_duration = _number(video.get("duration")) if video else None
    audio_start = _number(audio.get("start_time")) if audio else None
    audio_duration = _number(audio.get("duration")) if audio else None
    leading_video_gap = (
        max(0.0, video_start - timeline_start)
        if video_start is not None
        else None
    )
    video_coverage_ratio = (
        min(1.0, max(0.0, video_duration / timeline_duration))
        if video_duration is not None and timeline_duration > 0
        else None
    )
    return {
        "has_video": video is not None,
        "has_audio": audio is not None,
        "format_start_sec": format_start,
        "format_duration_sec": format_duration,
        "timeline_start_sec": timeline_start,
        "timeline_duration_sec": timeline_duration,
        "video_start_sec": video_start,
        "video_duration_sec": video_duration,
        "audio_start_sec": audio_start,
        "audio_duration_sec": audio_duration,
        "leading_video_gap_sec": leading_video_gap,
        "video_coverage_ratio": video_coverage_ratio,
    }


def probe_media_timing(
    path: Path,
    ffprobe: str = "ffprobe",
    *,
    timeout: int = 60,
) -> dict[str, Any]:
    resolved_ffprobe = shutil.which(ffprobe)
    if resolved_ffprobe is None and Path(ffprobe).name == ffprobe:
        sibling = Path(sys.executable).with_name(ffprobe)
        if sibling.is_file():
            resolved_ffprobe = str(sibling)
    if resolved_ffprobe is None:
        resolved_ffprobe = ffprobe
    result = subprocess.run(
        [
            resolved_ffprobe,
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
        timeout=timeout,
    )
    return parse_media_timing(json.loads(result.stdout))


def timing_flags(
    timing: dict[str, Any],
    *,
    max_leading_video_gap_sec: float = 0.5,
    min_video_coverage_ratio: float = 0.8,
) -> list[str]:
    flags = []
    if not timing["has_video"]:
        flags.append("no_video_stream")
        return flags
    gap = timing.get("leading_video_gap_sec")
    if gap is None:
        flags.append("unknown_video_start")
    elif gap > max_leading_video_gap_sec:
        flags.append("delayed_video_start")
    coverage = timing.get("video_coverage_ratio")
    if coverage is None:
        flags.append("unknown_video_coverage")
    elif coverage < min_video_coverage_ratio:
        flags.append("short_video_coverage")
    return flags
