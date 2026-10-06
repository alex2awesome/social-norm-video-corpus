#!/usr/bin/env python3
"""Render and register shadow-only bottom-crop instructional repairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from export_visual_audit_batch import extract_frames, probe_duration, sha256_json
from visual_audit_ledger import connect


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_stream_count(ffprobe: str, clip: Path) -> int:
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(clip),
        ],
        capture_output=True, text=True, check=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def video_dimensions(ffprobe: str, clip: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(clip),
        ],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream: {clip}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def resolve_crop_geometry(
    source_width: int,
    source_height: int,
    crop_bottom_fraction: float | None,
    crop_geometry: str | None,
) -> tuple[str, int, int]:
    """Return a validated ffmpeg crop geometry and its output dimensions."""
    if (crop_bottom_fraction is None) == (crop_geometry is None):
        raise ValueError("specify exactly one crop mode")
    if crop_geometry is None:
        assert crop_bottom_fraction is not None
        if not 0 < crop_bottom_fraction < 0.5:
            raise ValueError("crop-bottom-fraction must be between 0 and 0.5")
        output_height = math.floor(source_height * (1 - crop_bottom_fraction) / 2) * 2
        if output_height < 2:
            raise ValueError("crop produces an invalid output height")
        return f"{source_width}:{output_height}:0:0", source_width, output_height

    try:
        output_width, output_height, x_offset, y_offset = (
            int(part) for part in crop_geometry.split(":")
        )
    except (TypeError, ValueError):
        raise ValueError("crop-geometry must be width:height:x:y") from None
    if min(output_width, output_height) < 2 or min(x_offset, y_offset) < 0:
        raise ValueError("crop-geometry dimensions and offsets must be nonnegative")
    if output_width % 2 or output_height % 2:
        raise ValueError("crop-geometry width and height must be even for yuv420p output")
    if x_offset + output_width > source_width or y_offset + output_height > source_height:
        raise ValueError("crop-geometry exceeds the source frame")
    return crop_geometry, output_width, output_height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--item-id", action="append", required=True)
    crop_group = parser.add_mutually_exclusive_group(required=True)
    crop_group.add_argument("--crop-bottom-fraction", type=float)
    crop_group.add_argument(
        "--crop-geometry",
        help="explicit ffmpeg crop as width:height:x:y for spatially isolating a demo",
    )
    parser.add_argument("--strip-audio", action="store_true")
    parser.add_argument("--db", type=Path, default=Path("data/visual_audit/audit.db"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--model", default="gpt-5.6-sol")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    source = json.loads(args.source_manifest.read_text())
    source_items = {item["item_id"]: item for item in source["items"]}
    if set(args.item_id) - set(source_items):
        raise SystemExit(f"unknown items: {sorted(set(args.item_id) - set(source_items))}")

    args.out.mkdir(parents=True)
    frames_dir = args.out / "frames"
    clips_dir = args.out / "repaired_clips"
    frames_dir.mkdir()
    clips_dir.mkdir()
    project_root = args.project_root.resolve()
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed = sha256_json(
        {
            "item_ids": args.item_id,
            "crop_bottom_fraction": args.crop_bottom_fraction,
            "crop_geometry": args.crop_geometry,
            "strip_audio": args.strip_audio,
        }
    )
    batch_id = f"crop_repair_{stamp}_{hashlib.sha1(seed.encode()).hexdigest()[:8]}"
    manifest_path = args.out / "manifest.json"
    conn = connect(args.db)
    conn.execute(
        """
        INSERT INTO batches(batch_id,pillar,rubric_version,intended_model,
          selection_strategy,seed,manifest_path,created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            batch_id, "instructional", "instructional_repair_v1", args.model,
            "manual_spatial_crop_with_optional_audio_strip", seed,
            str(manifest_path.resolve()), time.time(),
        ),
    )
    records = []
    for ordinal, item_id in enumerate(args.item_id):
        item = source_items[item_id]
        source_clip = project_root / item["clip_path"]
        source_width, source_height = video_dimensions(args.ffprobe, source_clip)
        try:
            crop_geometry, output_width, output_height = resolve_crop_geometry(
                source_width,
                source_height,
                args.crop_bottom_fraction,
                args.crop_geometry,
            )
        except ValueError as exc:
            raise RuntimeError(f"invalid crop for {item_id}: {exc}") from exc
        repaired_clip = clips_dir / f"{ordinal:04d}.mp4"
        source_audio = audio_stream_count(args.ffprobe, source_clip)
        command = [
            args.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source_clip),
            "-map", "0:v:0", "-vf", f"crop={crop_geometry}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        ]
        if args.strip_audio:
            command += ["-an"]
        else:
            command += ["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"]
        command += ["-movflags", "+faststart", "-y", str(repaired_clip)]
        subprocess.run(command, check=True)
        repaired_audio = audio_stream_count(args.ffprobe, repaired_clip)
        if args.strip_audio and repaired_audio != 0:
            raise RuntimeError(f"audio remains after repair: {item_id}")
        if not args.strip_audio and repaired_audio != source_audio:
            raise RuntimeError(f"audio stream-count mismatch: {item_id}")
        source_duration = probe_duration(args.ffprobe, source_clip)
        duration, frames = extract_frames(
            args.ffmpeg, args.ffprobe, repaired_clip, frames_dir, ordinal, args.max_frames
        )
        if abs(duration - source_duration) > 0.35:
            raise RuntimeError(
                f"duration mismatch for {item_id}: {duration:.3f} vs {source_duration:.3f}"
            )
        transform = {
            "crop_geometry": crop_geometry,
            "crop_bottom_fraction": args.crop_bottom_fraction,
            "source_width": source_width,
            "source_height": source_height,
            "output_width": output_width,
            "output_height": output_height,
            "audio_removed": args.strip_audio,
            "source_audio_stream_count": source_audio,
            "repaired_audio_stream_count": repaired_audio,
        }
        record = {
            **{key: value for key, value in item.items() if key not in {"frames", "aligned_transcript"}},
            "ordinal": ordinal,
            "source_batch_id": source["batch_id"],
            "source_frame_manifest_sha256": item["frame_manifest_sha256"],
            "repair_type": "crop_label_overlay",
            "transform": transform,
            "repaired_clip_path": str(repaired_clip.relative_to(args.out)),
            "repaired_clip_sha256": sha256_file(repaired_clip),
            "duration": duration,
            "frames": frames,
            "aligned_transcript": [] if args.strip_audio else item.get("aligned_transcript", []),
            "source_label_transcript": item.get("aligned_transcript", []) if args.strip_audio else [],
        }
        record["frame_manifest_sha256"] = sha256_json(frames)
        records.append(record)
        conn.execute(
            "INSERT INTO batch_items(batch_id,item_id,ordinal,frame_count,frame_manifest_sha256) VALUES (?,?,?,?,?)",
            (batch_id, item_id, ordinal, len(frames), record["frame_manifest_sha256"]),
        )
    manifest = {
        "batch_id": batch_id,
        "pillar": "instructional",
        "rubric_version": "instructional_repair_v1",
        "intended_model": args.model,
        "source_manifest": str(args.source_manifest),
        "items": records,
        "created_at": time.time(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with (args.out / "manual_review.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "ordinal", "item_id", "decision", "candidate_start_sec",
                "candidate_end_sec", "polarity_supported", "norm_supported",
                "rejection_reasons", "manual_description",
            ]
        )
        for record in records:
            writer.writerow([record["ordinal"], record["item_id"], "", "", "", "", "", "", ""])
    conn.commit()
    conn.close()
    print(json.dumps({"batch_id": batch_id, "items": len(records), "frames": sum(len(r["frames"]) for r in records)}))


if __name__ == "__main__":
    main()
