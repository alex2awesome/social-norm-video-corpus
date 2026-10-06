#!/usr/bin/env python3
"""Render scalable witnessed candidate videos in the audited proxy format.

The frozen benchmark used silent video proxies with at most 96 frames over the
full stored clip, capped at 3 fps and 512 pixels, plus ASR text supplied to the
VLM.  This renderer reproduces that representation for localized candidate
windows.  Outputs are new shadow artifacts; source clips and metadata are never
modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


CLIP_SUFFIX = re.compile(r":clip_(\d+)$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_item_id(item_id: str) -> str:
    return CLIP_SUFFIX.sub(lambda match: f":{match.group(1)}", item_id)


def sampling_fps(duration: float, max_fps: float, max_frames: int) -> float:
    if duration <= 0 or max_fps <= 0 or max_frames < 1:
        raise ValueError("duration, max_fps, and max_frames must be positive")
    return min(max_fps, max_frames / duration)


def probe_timing(source: Path, ffprobe: str) -> tuple[float, float]:
    payload = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries", "format=start_time,duration",
        "-of", "json", str(source),
    ], text=True))
    fmt = payload.get("format") or {}
    start = max(0.0, float(fmt.get("start_time") or 0.0))
    duration = float(fmt["duration"]) - start
    if duration <= 0:
        raise ValueError(f"non-positive playable duration: {source}")
    return start, duration


def load_clip_segments(row: dict[str, Any]) -> list[dict[str, Any]]:
    transcript_path = Path(str(row.get("transcript_path") or ""))
    if not transcript_path.is_file():
        return []
    payload = json.loads(transcript_path.read_text())
    window = row.get("clip_window_source_sec") or [0, float("inf")]
    window_start, window_end = map(float, window)
    segments = []
    for segment in payload.get("segments") or []:
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        if end < window_start or start > window_end:
            continue
        segments.append({
            "text": str(segment.get("text") or "").strip(),
            "clip_start": start - window_start,
            "clip_end": end - window_start,
        })
    return segments


def source_from_candidate_parent(parent: dict[str, Any]) -> dict[str, Any] | None:
    """Recover a clip path when an older inventory manifest lacks the item."""
    metadata_value = str(parent.get("metadata_path") or "")
    clip_name = str(parent.get("clip_name") or "")
    if not metadata_value or not clip_name:
        return None
    clip = Path(metadata_value).parent / clip_name
    if not clip.is_file() or clip.stat().st_size <= 0:
        return None
    return {
        "item_id": canonical_item_id(str(parent["item_id"])),
        "source_clip": str(clip.resolve()),
        "source_recovery": "candidate_metadata_parent_plus_clip_name",
    }


def flatten_candidates(
    candidate_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    sources = {str(row["item_id"]): row for row in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("source manifest contains duplicate item_id")
    flattened, missing = [], []
    ordinal = 0
    for parent in candidate_rows:
        source = sources.get(canonical_item_id(str(parent["item_id"])))
        if source is None:
            source = source_from_candidate_parent(parent)
        if source is None:
            missing.append(str(parent["item_id"]))
            continue
        segments = load_clip_segments(parent)
        for candidate in parent.get("candidates") or []:
            if candidate.get("negative_self_defense") or candidate.get("negative_reported"):
                continue
            index = int(candidate["segment_index"])
            before = [row["text"] for row in segments[max(0, index - 4):index] if row["text"]]
            after = [row["text"] for row in segments[index + 1:index + 4] if row["text"]]
            candidate_id = f"{parent['item_id']}:candidate_{index}"
            explicit = (contexts or {}).get(candidate_id) or {}
            if explicit:
                before = explicit.get("context_before") or []
                after = explicit.get("context_after") or []
            flattened.append({
                "ordinal": ordinal,
                "candidate_id": candidate_id,
                "item_id": parent["item_id"],
                "uid": parent["uid"],
                "candidate_text": candidate["text"],
                "candidate_start_sec": float(candidate["start"]),
                "candidate_end_sec": float(candidate["end"]),
                "media_start_sec": float(candidate["window_start"]),
                "media_end_sec": float(candidate["window_end"]),
                "context_before": before,
                "context_after": after,
                "source_clip": source["source_clip"],
                "source_item_id": source["item_id"],
                "mechanisms": candidate.get("mechanisms") or [],
            })
            ordinal += 1
    return flattened, missing


def ffmpeg_argv(
    ffmpeg: str,
    source: Path,
    target: Path,
    *,
    source_start: float,
    window_start: float,
    window_duration: float,
    fps: float,
    max_width: int,
) -> list[str]:
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{source_start + window_start:.3f}", "-i", str(source),
        "-t", f"{window_duration:.3f}", "-an", "-vf",
        (
            f"fps={fps:.6f},"
            f"scale={max_width}:-2:force_original_aspect_ratio=decrease,"
            "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        ),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-movflags", "+faststart", str(target),
    ]


def full_proxy_ffmpeg_argv(
    ffmpeg: str,
    source: Path,
    target: Path,
    *,
    source_start: float,
    duration: float,
    fps: float,
    max_frames: int,
    max_width: int,
) -> list[str]:
    argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if source_start:
        argv.extend(["-ss", f"{source_start:.3f}"])
    argv.extend([
        "-i", str(source), "-t", f"{duration:.3f}", "-an", "-vf",
        (
            f"fps={fps:.6f},"
            f"scale={max_width}:-2:force_original_aspect_ratio=decrease,"
            "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        ),
        "-frames:v", str(max_frames), "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "30", "-movflags", "+faststart", str(target),
    ])
    return argv


def cut_proxy_ffmpeg_argv(
    ffmpeg: str,
    source: Path,
    target: Path,
    *,
    window_start: float,
    window_duration: float,
) -> list[str]:
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(source), "-ss", f"{window_start:.3f}",
        "-t", f"{window_duration:.3f}", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "27", "-movflags", "+faststart",
        str(target),
    ]


def render_full_proxy(
    row: dict[str, Any],
    proxy_dir: Path,
    ffmpeg: str,
    ffprobe: str,
    max_fps: float,
    max_frames: int,
    max_width: int,
) -> dict[str, Any]:
    source = Path(row["source_clip"])
    digest = hashlib.sha256(row["source_item_id"].encode()).hexdigest()[:16]
    target = proxy_dir / f"{digest}.mp4"
    try:
        if not source.is_file():
            raise FileNotFoundError(source)
        source_start, source_duration = probe_timing(source, ffprobe)
        fps = sampling_fps(source_duration, max_fps, max_frames)
        proxy_dir.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size == 0:
            subprocess.run(full_proxy_ffmpeg_argv(
                ffmpeg, source, target, source_start=source_start,
                duration=source_duration, fps=fps, max_frames=max_frames,
                max_width=max_width,
            ), check=True)
        _proxy_start, proxy_duration = probe_timing(target, ffprobe)
        return {
            "source_item_id": row["source_item_id"],
            "proxy_clip": str(target.resolve()),
            "proxy_sha256": sha256(target),
            "source_duration_sec": source_duration,
            "proxy_duration_sec": proxy_duration,
            "sampling_fps": fps,
            "error": None,
        }
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {
            "source_item_id": row["source_item_id"],
            "proxy_clip": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def render_one(
    row: dict[str, Any],
    output_dir: Path,
    portable_root: Path,
    ffmpeg: str,
    ffprobe: str,
    max_fps: float,
    max_frames: int,
    max_width: int,
    proxy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(proxy["proxy_clip"] if proxy else row["source_clip"])
    digest = hashlib.sha256(row["candidate_id"].encode()).hexdigest()[:16]
    target = output_dir / f"{row['ordinal']:06d}_{digest}.mp4"
    try:
        if not source.is_file():
            raise FileNotFoundError(source)
        source_start, playable_duration = probe_timing(source, ffprobe)
        source_duration = (
            float(proxy["source_duration_sec"]) if proxy else playable_duration
        )
        start = min(max(0.0, row["media_start_sec"]), max(0.0, source_duration - 0.05))
        end = min(source_duration, max(start + 0.05, row["media_end_sec"]))
        fps = (
            float(proxy["sampling_fps"])
            if proxy else sampling_fps(source_duration, max_fps, max_frames)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size == 0:
            argv = (
                cut_proxy_ffmpeg_argv(
                    ffmpeg, source, target, window_start=start,
                    window_duration=end - start,
                )
                if proxy
                else ffmpeg_argv(
                    ffmpeg, source, target, source_start=source_start,
                    window_start=start, window_duration=end - start,
                    fps=fps, max_width=max_width,
                )
            )
            subprocess.run(argv, check=True)
        candidate_start = min(end - start, max(0.0, row["candidate_start_sec"] - start))
        candidate_end = min(end - start, max(candidate_start, row["candidate_end_sec"] - start))
        try:
            media_path = target.resolve().relative_to(portable_root.resolve())
        except ValueError:
            media_path = target.resolve()
        return {
            **row,
            "candidate_video_path": str(media_path),
            "candidate_video_sha256": sha256(target),
            "candidate_relative_start_sec": candidate_start,
            "candidate_relative_end_sec": candidate_end,
            "window_duration_sec": end - start,
            "source_duration_sec": source_duration,
            "full_proxy_clip": proxy.get("proxy_clip") if proxy else None,
            "full_proxy_sha256": proxy.get("proxy_sha256") if proxy else None,
            "two_stage_full_proxy_then_candidate_cut": proxy is not None,
            "sampling_fps": fps,
            "max_source_frames": max_frames,
            "max_width": max_width,
            "error": None,
            "policy": "shadow_media_only_preserve_source_and_metadata",
        }
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {
            **row,
            "candidate_video_path": None,
            "candidate_video_sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
            "policy": "shadow_media_only_preserve_source_and_metadata",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--contexts", type=Path,
        help="optional candidate-keyed JSONL supplying frozen ASR context",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--max-fps", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--proxy-dir", type=Path,
        help="enable audited two-stage full-clip proxy then candidate-cut rendering",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    for path in (args.out_manifest, args.failures, args.summary):
        if path.exists():
            raise SystemExit(f"refusing to overwrite: {path}")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index/count")
    contexts = None
    if args.contexts:
        context_rows = read_jsonl(args.contexts)
        contexts = {str(row["candidate_id"]): row for row in context_rows}
        if len(contexts) != len(context_rows):
            raise SystemExit("duplicate candidate_id in --contexts")
    rows, missing = flatten_candidates(
        read_jsonl(args.candidates), read_jsonl(args.source_manifest), contexts
    )
    selected = [row for row in rows if row["ordinal"] % args.shard_count == args.shard_index]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    proxies: dict[str, dict[str, Any]] = {}
    if args.proxy_dir:
        unique = {row["source_item_id"]: row for row in selected}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    render_full_proxy, row, args.proxy_dir, args.ffmpeg,
                    args.ffprobe, args.max_fps, args.max_frames, args.max_width,
                ): item_id for item_id, row in unique.items()
            }
            for future in as_completed(futures):
                result = future.result()
                proxies[result["source_item_id"]] = result
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                render_one, row, args.out_dir, args.out_manifest.parent,
                args.ffmpeg, args.ffprobe, args.max_fps, args.max_frames,
                args.max_width,
                proxies.get(row["source_item_id"]) if args.proxy_dir else None,
            ): row for row in selected
            if not args.proxy_dir
            or proxies.get(row["source_item_id"], {}).get("error") is None
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["ordinal"])
    successes = [row for row in results if row["error"] is None]
    failures = [row for row in results if row["error"] is not None]
    proxy_failures = [row for row in proxies.values() if row.get("error")]
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in successes))
    args.failures.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in failures))
    summary = {
        "kind": "witnessed_reaction_video_asr_windows_v1",
        "candidate_rows": len(rows),
        "selected_for_shard": len(selected),
        "rendered": len(successes),
        "failed": len(failures),
        "full_proxies": len(proxies) - len(proxy_failures),
        "full_proxy_failures": len(proxy_failures),
        "missing_source_items": len(missing),
        "missing_source_item_examples": missing[:20],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "max_fps": args.max_fps,
        "max_frames_over_full_source_clip": args.max_frames,
        "max_width": args.max_width,
        "audio_included": False,
        "two_stage_full_proxy_then_candidate_cut": bool(args.proxy_dir),
        "manifest_sha256": sha256(args.out_manifest),
        "failures_sha256": sha256(args.failures),
        "corpus_mutated": False,
        "policy": "shadow_only_no_keep_reject_or_source_mutation",
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures and not proxy_failures and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
