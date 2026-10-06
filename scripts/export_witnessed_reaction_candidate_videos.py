#!/usr/bin/env python3
"""Materialize small audiovisual windows for witnessed reaction candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_window(
    row: dict[str, Any], source_duration: float | None = None
) -> tuple[float, float, float, float]:
    start = max(0.0, float(row["media_start_sec"]))
    end = max(start + 0.05, float(row["media_end_sec"]))
    if source_duration is not None:
        end = min(end, source_duration)
        start = min(start, max(0.0, end - 0.05))
    candidate_start = min(end - start, max(0.0, float(row["candidate_start_sec"]) - start))
    candidate_end = min(end - start, max(candidate_start, float(row["candidate_end_sec"]) - start))
    return start, end, candidate_start, candidate_end


def resolve_media(row: dict[str, Any], manifest: Path) -> Path:
    source = Path(row["proxy_clip"])
    candidates = [source, manifest.parent / source]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"missing proxy clip for {row['candidate_id']}: {source}")


def ffprobe_duration(ffprobe: str, source: Path) -> float:
    value = subprocess.check_output(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(source)],
        text=True,
    ).strip()
    duration = float(value)
    if duration <= 0:
        raise ValueError(f"non-positive duration for {source}")
    return duration


def ffmpeg_argv(
    ffmpeg: str,
    source: Path,
    target: Path,
    start: float,
    end: float,
    *,
    fps: float | None = None,
    max_width: int | None = None,
) -> list[str]:
    argv = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a?",
    ]
    filters = []
    if fps is not None:
        filters.append(f"fps={fps:g}")
    if max_width is not None:
        filters.append(
            f"scale={max_width}:-2:force_original_aspect_ratio=decrease"
        )
    if filters:
        argv.extend(["-vf", ",".join(filters)])
    argv.extend([
        "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "27", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
        str(target),
    ])
    return argv


def export(
    rows: list[dict[str, Any]],
    manifest: Path,
    out_dir: Path,
    ffmpeg: str,
    contexts: dict[str, dict[str, Any]] | None = None,
    ffprobe: str = "ffprobe",
    fps: float | None = None,
    max_width: int | None = None,
    portable_root: Path | None = None,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for index, row in enumerate(rows):
        source = resolve_media(row, manifest)
        source_duration = ffprobe_duration(ffprobe, source)
        start, end, candidate_start, candidate_end = bounded_window(row, source_duration)
        target = out_dir / f"{index:04d}.mp4"
        if not target.exists():
            subprocess.run(
                ffmpeg_argv(
                    ffmpeg, source, target, start, end,
                    fps=fps, max_width=max_width,
                ),
                check=True,
            )
        context = (contexts or {}).get(str(row["candidate_id"]), {})
        try:
            portable_target = target.resolve().relative_to(
                (portable_root or manifest.parent).resolve()
            )
        except ValueError:
            portable_target = target.resolve()
        exported.append({
            **row,
            "context_before": context.get("context_before") or [],
            "context_after": context.get("context_after") or [],
            "candidate_video_path": str(portable_target),
            "candidate_video_sha256": sha256(target),
            "window_start_sec": start,
            "window_end_sec": end,
            "candidate_relative_start_sec": candidate_start,
            "candidate_relative_end_sec": candidate_end,
            "window_duration_sec": end - start,
            "source_duration_sec": source_duration,
            "encoding_fps": fps,
            "encoding_max_width": max_width,
            "policy": "audit_media_only_no_corpus_mutation",
        })
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument(
        "--contexts",
        type=Path,
        help="optional candidate-keyed JSONL containing context_before/context_after",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--max-width", type=int)
    args = parser.parse_args()
    if args.out_manifest.exists():
        raise SystemExit(f"refusing to overwrite: {args.out_manifest}")
    if args.fps is not None and args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.max_width is not None and args.max_width < 64:
        raise SystemExit("--max-width must be at least 64")
    contexts = None
    if args.contexts:
        context_rows = read_jsonl(args.contexts)
        contexts = {str(row["candidate_id"]): row for row in context_rows}
        if len(contexts) != len(context_rows):
            raise SystemExit("duplicate candidate_id in --contexts")
    rows = export(
        read_jsonl(args.manifest), args.manifest, args.out_dir, args.ffmpeg, contexts,
        args.ffprobe,
        args.fps,
        args.max_width,
        args.out_manifest.parent,
    )
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(json.dumps({
        "kind": "witnessed_reaction_candidate_video_windows_v1",
        "candidates": len(rows),
        "manifest_sha256": sha256(args.out_manifest),
        "corpus_mutated": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
