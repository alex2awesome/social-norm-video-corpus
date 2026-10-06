#!/usr/bin/env python3
"""Render blinded, original-audio media for a witnessed corpus audit.

Model scores, cohort names, ASR text, and weak labels are omitted from the
manual-review manifest.  Each selected candidate is recut from its original
source clip with audio retained and denser video than the model proxy.  Source
media and corpus metadata are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2

if __package__:
    from scripts.render_dense_blind_followup import make_dense_sheet
    from scripts.score_visual_scene_baselines import sample_frames
else:
    from render_dense_blind_followup import make_dense_sheet
    from score_visual_scene_baselines import sample_frames


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    audit_index = 0
    for clip in rows:
        for candidate in clip.get("candidates") or []:
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id or candidate_id in seen:
                raise ValueError(f"empty or duplicate candidate_id: {candidate_id!r}")
            seen.add(candidate_id)
            source = str(candidate.get("source_clip") or "")
            start = float(candidate.get("media_start_sec") or 0)
            end = float(candidate.get("media_end_sec") or 0)
            if not source or end <= start:
                raise ValueError(f"{candidate_id}: invalid source or media interval")
            output.append({
                "audit_candidate_index": audit_index,
                "clip_audit_index": int(clip["clip_audit_index"]),
                "candidate_id": candidate_id,
                "item_id": str(clip["item_id"]),
                "uid": str(clip["uid"]),
                "source_clip": source,
                "media_start_sec": start,
                "media_end_sec": end,
                "candidate_relative_start_sec": float(
                    candidate.get("candidate_relative_start_sec") or 0
                ),
                "candidate_relative_end_sec": float(
                    candidate.get("candidate_relative_end_sec") or 0
                ),
            })
            audit_index += 1
    return output


def ffmpeg_argv(ffmpeg: str, row: dict[str, Any], target: Path) -> list[str]:
    duration = row["media_end_sec"] - row["media_start_sec"]
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{row['media_start_sec']:.3f}", "-i", row["source_clip"],
        "-t", f"{duration:.3f}",
        "-vf", (
            "fps=12,scale=720:-2:force_original_aspect_ratio=decrease,"
            "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        ),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
        str(target),
    ]


def probe(ffprobe: str, path: Path) -> dict[str, Any]:
    payload = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries",
        "format=duration:stream=codec_type", "-of", "json", str(path),
    ], text=True))
    streams = payload.get("streams") or []
    return {
        "duration_sec": float((payload.get("format") or {}).get("duration") or 0),
        "audio_present": any(row.get("codec_type") == "audio" for row in streams),
        "video_present": any(row.get("codec_type") == "video" for row in streams),
    }


def render_storyboard(
    media_path: Path,
    storyboard_path: Path,
    *,
    audit_candidate_index: int,
    requested_frames: int,
) -> dict[str, Any]:
    """Render a dense, prediction-blind overview of the complete audit clip."""
    if requested_frames < 12 or requested_frames % 12:
        raise ValueError("storyboard frame count must be a positive multiple of 12")
    frames, metadata = sample_frames(media_path, requested_frames)
    timestamps = metadata.get("sampled_timestamps") or []
    if not frames or len(frames) != len(timestamps):
        raise ValueError("storyboard sampling produced no aligned frames")
    sheet = make_dense_sheet(frames, timestamps, audit_candidate_index)
    storyboard_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(storyboard_path), sheet):
        raise OSError(f"failed to write storyboard: {storyboard_path}")
    return {
        "storyboard_path": str(storyboard_path.resolve()),
        "storyboard_sha256": sha256(storyboard_path),
        "storyboard_requested_frames": requested_frames,
        "storyboard_rendered_frames": len(frames),
        "storyboard_sampled_timestamps_sec": timestamps,
    }


def render_one(
    row: dict[str, Any], output_dir: Path, ffmpeg: str, ffprobe: str,
    storyboard_frames: int,
) -> dict[str, Any]:
    digest = hashlib.sha256(row["candidate_id"].encode()).hexdigest()[:16]
    target = output_dir / f"{row['audit_candidate_index']:04d}_{digest}.mp4"
    storyboard = output_dir / "storyboards" / (
        f"{row['audit_candidate_index']:04d}_{digest}.jpg"
    )
    try:
        source = Path(row["source_clip"])
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.stat().st_size <= 0:
            subprocess.run(ffmpeg_argv(ffmpeg, row, target), check=True)
        media = probe(ffprobe, target)
        if not media["video_present"] or media["duration_sec"] <= 0:
            raise ValueError("rendered media has no playable video")
        storyboard_metadata = render_storyboard(
            target,
            storyboard,
            audit_candidate_index=row["audit_candidate_index"],
            requested_frames=storyboard_frames,
        )
        return {
            "audit_candidate_index": row["audit_candidate_index"],
            "clip_audit_index": row["clip_audit_index"],
            "candidate_id": row["candidate_id"],
            "item_id": row["item_id"],
            "uid": row["uid"],
            "manual_media_path": str(target.resolve()),
            "manual_media_sha256": sha256(target),
            "duration_sec": media["duration_sec"],
            "audio_present": media["audio_present"],
            "video_present": media["video_present"],
            "candidate_relative_start_sec": row["candidate_relative_start_sec"],
            "candidate_relative_end_sec": row["candidate_relative_end_sec"],
            "error": None,
            "policy": "blind_manual_audit_only_no_corpus_mutation",
            **storyboard_metadata,
        }
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {
            "audit_candidate_index": row["audit_candidate_index"],
            "clip_audit_index": row["clip_audit_index"],
            "candidate_id": row["candidate_id"],
            "item_id": row["item_id"],
            "uid": row["uid"],
            "manual_media_path": None,
            "manual_media_sha256": None,
            "storyboard_path": None,
            "storyboard_sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
            "policy": "blind_manual_audit_only_no_corpus_mutation",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--storyboard-frames", type=int, default=24)
    args = parser.parse_args()
    for path in (args.manifest, args.failures, args.summary):
        if path.exists():
            raise SystemExit(f"refusing to overwrite: {path}")
    rows = flatten_selection(read_jsonl(args.selection))
    if args.storyboard_frames < 12 or args.storyboard_frames % 12:
        raise SystemExit("--storyboard-frames must be a positive multiple of 12")
    results = [
        render_one(
            row, args.out_dir, args.ffmpeg, args.ffprobe,
            args.storyboard_frames,
        )
        for row in rows
    ]
    successes = [row for row in results if row["error"] is None]
    failures = [row for row in results if row["error"] is not None]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in successes))
    args.failures.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in failures))
    summary = {
        "kind": "witnessed_corpus_blind_manual_media_v1",
        "selected_candidates": len(rows),
        "rendered": len(successes),
        "failed": len(failures),
        "with_audio": sum(bool(row.get("audio_present")) for row in successes),
        "with_storyboards": sum(bool(row.get("storyboard_path")) for row in successes),
        "manifest_sha256": sha256(args.manifest),
        "failures_sha256": sha256(args.failures),
        "model_outputs_exposed_in_manifest": False,
        "corpus_mutated": False,
        "policy": "blind_manual_audit_only_no_keep_reject_or_corpus_mutation",
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
