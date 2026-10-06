#!/usr/bin/env python3
"""Render exact, hashed repair candidates for manually reviewed shadow outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.export_instructional_crop_repair_audit import (
    audio_stream_count,
    resolve_crop_geometry,
    sha256_file,
    video_dimensions,
)
from scripts.export_visual_audit_batch import extract_frames, probe_duration, sha256_json


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("repair_specs", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--kind", default="instructional_shadow_exact_repairs")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    source = load_json(args.source_manifest)
    specs_value = load_json(args.repair_specs)
    specs = specs_value.get("repairs") or []
    if not isinstance(specs, list) or not specs:
        raise SystemExit("repair_specs.repairs must be a nonempty list")
    source_items = {int(item["ordinal"]): item for item in source.get("items") or []}
    spec_ordinals = [int(spec["source_ordinal"]) for spec in specs]
    if len(set(spec_ordinals)) != len(spec_ordinals):
        raise SystemExit("source_ordinal values must be unique")
    if set(spec_ordinals) - set(source_items):
        raise SystemExit(f"unknown source ordinals: {sorted(set(spec_ordinals) - set(source_items))}")

    root = args.project_root.resolve()
    args.out.mkdir(parents=True)
    clips_dir = args.out / "repaired_clips"
    frames_dir = args.out / "frames"
    clips_dir.mkdir()
    frames_dir.mkdir()
    records = []
    for ordinal, spec in enumerate(specs):
        source_ordinal = int(spec["source_ordinal"])
        source_item = source_items[source_ordinal]
        raw = root / source_item["source_raw_path"]
        start = float(spec["source_start_sec"])
        end = float(spec["source_end_sec"])
        raw_duration = probe_duration(args.ffprobe, raw)
        if not 0 <= start < end <= raw_duration:
            raise SystemExit(f"invalid raw interval for source ordinal {source_ordinal}")
        source_width, source_height = video_dimensions(args.ffprobe, raw)
        crop_geometry = spec.get("crop_geometry")
        if crop_geometry:
            crop_geometry, output_width, output_height = resolve_crop_geometry(
                source_width, source_height, None, str(crop_geometry)
            )
            video_filter = f"crop={crop_geometry},scale='min(640,iw)':-2"
        else:
            output_width = min(640, source_width)
            output_height = None
            video_filter = "scale='min(640,iw)':-2"
        strip_audio = bool(spec.get("strip_audio"))
        repaired = clips_dir / f"{ordinal:04d}.mp4"
        seek_after_input = bool(spec.get("seek_after_input"))
        command = [args.ffmpeg, "-hide_banner", "-loglevel", "error"]
        if not seek_after_input:
            command += ["-ss", f"{start:.3f}"]
        command += ["-i", str(raw)]
        if seek_after_input:
            command += ["-ss", f"{start:.3f}"]
        command += [
            "-t", f"{end - start:.3f}", "-map", "0:v:0", "-vf", video_filter,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        ]
        if strip_audio:
            command += ["-an"]
        else:
            command += ["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"]
        command += ["-movflags", "+faststart", "-y", str(repaired)]
        subprocess.run(command, check=True)
        repaired_duration = probe_duration(args.ffprobe, repaired)
        if abs(repaired_duration - (end - start)) > 0.35:
            raise RuntimeError(
                f"duration mismatch for source ordinal {source_ordinal}: "
                f"{repaired_duration:.3f} vs {end-start:.3f}"
            )
        repaired_audio = audio_stream_count(args.ffprobe, repaired)
        if strip_audio and repaired_audio:
            raise RuntimeError(f"audio remains for source ordinal {source_ordinal}")
        _duration, frames = extract_frames(
            args.ffmpeg, args.ffprobe, repaired, frames_dir, ordinal, args.max_frames
        )
        record = {
            "ordinal": ordinal,
            "source_ordinal": source_ordinal,
            "uid": source_item["uid"],
            "shadow_index": source_item["shadow_index"],
            "source_manifest_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
            "source_item_clip_sha256": source_item.get("clip_sha256"),
            "source_raw_path": source_item["source_raw_path"],
            "source_start_sec": start,
            "source_end_sec": end,
            "requested_duration_sec": end - start,
            "rendered_duration_sec": repaired_duration,
            "repair_reason": spec.get("reason"),
            "proposed_polarity": spec.get("proposed_polarity"),
            "proposed_behavior": spec.get("proposed_behavior"),
            "proposed_norm": spec.get("proposed_norm"),
            "transform": {
                "crop_geometry": crop_geometry,
                "source_width": source_width,
                "source_height": source_height,
                "pre_scale_crop_width": output_width,
                "pre_scale_crop_height": output_height,
                "audio_removed": strip_audio,
                "seek_after_input": seek_after_input,
                "repaired_audio_stream_count": repaired_audio,
            },
            "repaired_clip_path": str(repaired.relative_to(args.out)),
            "repaired_clip_sha256": sha256_file(repaired),
            "frames": frames,
        }
        record["repaired_frame_manifest_sha256"] = sha256_json(frames)
        records.append(record)

    manifest = {
        "kind": args.kind,
        "source_manifest": str(args.source_manifest),
        "repair_specs": str(args.repair_specs),
        "corpus_mutated": False,
        "items": records,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.out / "manual_review.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "ordinal", "source_ordinal", "uid", "decision", "social_norm", "visual_demo",
            "polarity_supported", "norm_supported", "interval_tight", "label_leak_visible",
            "narration_leak", "rejection_reasons", "manual_description",
        ])
        for record in records:
            writer.writerow([record["ordinal"], record["source_ordinal"], record["uid"]] + [""] * 10)
    print(json.dumps({"repairs": len(records), "frames": sum(len(r["frames"]) for r in records)}))


if __name__ == "__main__":
    main()
