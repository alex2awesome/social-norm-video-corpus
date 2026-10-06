#!/usr/bin/env python3
"""Materialize muted, cropped commentary repair candidates for final audit.

The source clips and corpus metadata are read-only. Every output remains
explicitly unapproved until a separate post-transform manual ledger is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2

try:
    from src.media_integrity import probe_media_timing, timing_flags
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.media_integrity import probe_media_timing, timing_flags

if __package__:
    from scripts.render_dense_blind_followup import make_dense_sheet
    from scripts.score_visual_scene_baselines import sample_frames
else:
    from render_dense_blind_followup import make_dense_sheet
    from score_visual_scene_baselines import sample_frames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def validate_plan(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty repair plan")
    indices: set[int] = set()
    variants: set[str] = set()
    for row in rows:
        index = int(row["audit_index"])
        variant = str(row["variant_id"])
        if index in indices or variant in variants:
            raise ValueError(f"duplicate repair target: {index}/{variant}")
        indices.add(index)
        variants.add(variant)
        if row.get("approval_status") != "unreviewed_repair_candidate":
            raise ValueError(f"repair is incorrectly pre-approved: {index}")
        source = Path(row["source_path"])
        if not source.is_file():
            raise ValueError(f"missing source: {source}")
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        if not 0 <= start < end:
            raise ValueError(f"invalid time bounds: {index}")
        crop = row.get("crop_norm")
        if not isinstance(crop, list) or len(crop) != 4:
            raise ValueError(f"invalid crop: {index}")
        x0, y0, x1, y1 = map(float, crop)
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"invalid crop bounds: {index}")


def crop_filter(crop: list[float]) -> str:
    x0, y0, x1, y1 = map(float, crop)
    width = x1 - x0
    height = y1 - y0
    # Force even dimensions and offsets for yuv420p/libx264.
    return (
        f"crop="
        f"w=trunc(iw*{width:.8f}/2)*2:"
        f"h=trunc(ih*{height:.8f}/2)*2:"
        f"x=trunc(iw*{x0:.8f}/2)*2:"
        f"y=trunc(ih*{y0:.8f}/2)*2"
    )


def make_repair_sheet(
    frames: list,
    timestamps: list[float | None],
    audit_index: int,
):
    """Render every sampled frame, in stacked 12-frame panels."""
    return make_dense_sheet(frames, timestamps, audit_index)


def render_repair(
    row: dict,
    target: Path,
    *,
    ffmpeg: str,
    ffprobe: str,
) -> dict:
    start = float(row["start_sec"])
    end = float(row["end_sec"])
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(row["source_path"]),
            "-t",
            f"{end - start:.3f}",
            "-vf",
            crop_filter(row["crop_norm"]),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    timing = probe_media_timing(target, ffprobe)
    flags = timing_flags(timing)
    if timing["has_audio"]:
        flags.append("unexpected_audio")
    if flags:
        raise RuntimeError(f"media integrity flags: {flags}")
    return timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite repair directory: {args.out}")
    rows = load_jsonl(args.plan)
    validate_plan(rows)

    args.out.mkdir(parents=True)
    clips = args.out / "clips"
    sheets = args.out / "blind_sheets"
    clips.mkdir()
    sheets.mkdir()
    rendered = []
    failures = []
    for row in rows:
        index = int(row["audit_index"])
        target = clips / f"{row['variant_id']}.mp4"
        try:
            timing = render_repair(
                row, target, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
            )
            frames, media = sample_frames(target, 24)
            duration = float(row["end_sec"]) - float(row["start_sec"])
            minimum = min(8, max(4, int(duration * 3) - 1))
            if len(frames) < minimum:
                raise RuntimeError(
                    f"repair produced {len(frames)} audit frames; need {minimum}"
                )
            sheet = make_repair_sheet(
                frames, media["sampled_timestamps"], index
            )
            sheet_path = sheets / f"{row['variant_id']}.jpg"
            if not cv2.imwrite(
                str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 93]
            ):
                raise RuntimeError(f"failed to write {sheet_path}")
        except Exception as exc:
            failures.append(
                {
                    **row,
                    "repair_clip": str(target) if target.exists() else None,
                    "repair_clip_sha256": (
                        sha256(target) if target.exists() else None
                    ),
                    "approval_status": "render_failed",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        rendered.append(
            {
                **row,
                "repair_clip": str(target),
                "repair_clip_sha256": sha256(target),
                "sheet_path": str(sheet_path),
                "sheet_sha256": sha256(sheet_path),
                "transform": "temporal_cut_normalized_crop_and_mute",
                "audio_preserved": False,
                "media_timing": timing,
                "approval_status": "awaiting_post_transform_audit",
            }
        )

    manifest = args.out / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rendered)
    )
    failure_path = args.out / "failures.jsonl"
    failure_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures)
    )
    summary = {
        "kind": "commentary_spatial_repair_candidates",
        "planned": len(rows),
        "rendered": len(rendered),
        "failed": len(failures),
        "approval_status": "awaiting_post_transform_audit",
        "audio_preserved": False,
        "source_mutated": False,
        "metadata_mutated": False,
        "plan_sha256": sha256(args.plan),
        "manifest_sha256": sha256(manifest),
        "failures_sha256": sha256(failure_path),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
