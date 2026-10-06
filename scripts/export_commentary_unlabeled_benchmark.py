#!/usr/bin/env python3
"""Cut transcript-centered commentary targets without assigning gold labels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from src.media_integrity import probe_media_timing, timing_flags
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.media_integrity import probe_media_timing, timing_flags


VIDEO_SUFFIXES = ("mp4", "webm", "mkv", "mov", "m4v")


def requested_duration_flags(
    timing: dict, expected_duration: float, minimum_coverage: float = 0.90
) -> list[str]:
    """Reject internally valid outputs that cover too little of the requested cut."""
    actual = float(timing.get("video_duration_sec") or 0)
    if expected_duration <= 0:
        return ["invalid_requested_duration"]
    coverage = actual / expected_duration
    return [] if coverage >= minimum_coverage else [
        f"requested_duration_coverage={coverage:.6f}"
    ]


def audio_output_args(strip_audio: bool) -> list[str]:
    return ["-an"] if strip_audio else ["-c:a", "aac", "-b:a", "96k"]


def composed_video_filter(spatial_filter: str | None = None) -> str:
    """Build the fixed low-rate audit proxy filter without shell interpolation."""
    filters = ["fps=3"]
    if spatial_filter is not None:
        spatial_filter = spatial_filter.strip()
        if not spatial_filter or any(value in spatial_filter for value in (";", "[", "]")):
            raise ValueError("invalid spatial filter")
        filters.append(spatial_filter)
    filters.append("scale='min(640,iw)':-2")
    return ",".join(filters)


def resolve_source(video_root: Path, uid: str) -> Path:
    matches = [
        path
        for suffix in VIDEO_SUFFIXES
        if (path := video_root / f"{uid}.{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one source for {uid}, found {len(matches)}")
    return matches[0]


def seek_input_args(source: Path, start: float, mode: str) -> list[str]:
    if mode == "fast":
        return ["-ss", f"{start:.3f}", "-i", str(source)]
    if mode == "hybrid_30s":
        coarse = max(0.0, start - 30.0)
        fine = start - coarse
        return [
            "-ss",
            f"{coarse:.3f}",
            "-i",
            str(source),
            "-ss",
            f"{fine:.3f}",
        ]
    if mode == "accurate":
        return ["-i", str(source), "-ss", f"{start:.3f}"]
    raise ValueError(f"unknown seek mode: {mode}")


def render_verified_target(
    source: Path,
    target: Path,
    start: float,
    end: float,
    *,
    ffmpeg: str,
    ffprobe: str,
    strip_audio: bool,
    spatial_filter: str | None = None,
) -> tuple[str, dict]:
    errors = []
    audio_args = audio_output_args(strip_audio)
    for mode in ("fast", "hybrid_30s", "accurate"):
        target.unlink(missing_ok=True)
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    *seek_input_args(source, start, mode),
                    "-t",
                    f"{end - start:.3f}",
                    "-vf",
                    composed_video_filter(spatial_filter),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "28",
                    *audio_args,
                    "-movflags",
                    "+faststart",
                    "-y",
                    str(target),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            timing = probe_media_timing(target, ffprobe)
            flags = timing_flags(timing) + requested_duration_flags(
                timing, end - start
            )
            if not flags:
                return mode, timing
            errors.append(f"{mode}: timing flags {flags}")
        except (
            subprocess.CalledProcessError,
            subprocess.SubprocessError,
            ValueError,
        ) as exc:
            errors.append(f"{mode}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered-manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--strip-audio",
        action="store_true",
        help="Explicitly discard audio; by default compact clips retain AAC audio.",
    )
    args = parser.parse_args()

    rendered = json.loads(args.rendered_manifest.read_text())["records"]
    args.out.mkdir(parents=True, exist_ok=True)
    clips = args.out / "clips"
    clips.mkdir(exist_ok=True)
    output = []
    failures = []
    for row in rendered:
        start, end = row["target_window"]
        try:
            source = resolve_source(args.video_root, row["uid"])
        except ValueError as exc:
            failures.append({"uid": row["uid"], "error": str(exc)})
            continue
        target = clips / f"{int(row['ordinal']):02d}__{row['uid']}.mp4"
        try:
            seek_mode, timing = render_verified_target(
                source,
                target,
                start,
                end,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                strip_audio=args.strip_audio,
            )
        except RuntimeError as exc:
            failures.append({"uid": row["uid"], "error": str(exc)})
            continue
        output.append(
            {
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": "commentary",
                "title": row.get("title"),
                "category": row.get("category"),
                "norm": row.get("normalized_norm"),
                "normalized_behavior": row.get("normalized_behavior"),
                "explanation": row.get("behavior_evidence_quote"),
                "start_quote": row.get("behavior_evidence_quote"),
                "end_quote": row.get("behavior_evidence_quote"),
                "source_platform": row.get("source_platform"),
                "query_source": row.get("query_source"),
                "signal": row.get("signal"),
                "proxy_clip": str(target.resolve()),
                "target_window": row["target_window"],
                "audio_preserved": not args.strip_audio,
                "seek_mode": seek_mode,
                "media_timing": timing,
            }
        )
    manifest = args.out / "manifest.jsonl"
    with manifest.open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "items": len(output),
        "failures": failures,
        "gold_assigned": False,
        "audio_preserved": not args.strip_audio,
        "manifest": str(manifest),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
