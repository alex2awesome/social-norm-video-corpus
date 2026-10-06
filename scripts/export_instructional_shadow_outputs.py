#!/usr/bin/env python3
"""Render every shadow instructional span for mandatory manual visual review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from src.batch_transcribe import _find_raw
from scripts.export_instructional_crop_repair_audit import sha256_file
from scripts.export_visual_audit_batch import extract_frames, probe_duration, sha256_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compare", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shadow-name", default=None)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    root = args.project_root.resolve()
    cfg = yaml.safe_load((root / "config/settings.yaml").read_text())
    comparison: dict[str, Any] = json.loads(args.compare.read_text())
    shadow_name = args.shadow_name or str(comparison.get("shadow_name") or "shadow_v2")
    args.out.mkdir(parents=True)
    clips_dir = args.out / "clips"
    frames_dir = args.out / "frames"
    clips_dir.mkdir()
    frames_dir.mkdir()

    items = []
    ordinal = 0
    for record in comparison.get("records") or []:
        raw = _find_raw(cfg, str(record["uid"]))
        for shadow_index, demo in enumerate((record.get(shadow_name) or {}).get("demos") or []):
            start = demo.get("start_sec")
            end = demo.get("end_sec")
            if raw is None or start is None or end is None or float(end) <= float(start):
                item = {
                    "ordinal": ordinal,
                    "uid": record["uid"],
                    "shadow_index": shadow_index,
                    "demo": demo,
                    "render_error": "raw missing or invalid/ungrounded interval",
                }
                items.append(item)
                ordinal += 1
                continue
            start = float(start)
            end = float(end)
            clip = clips_dir / f"{ordinal:03d}_{record['uid']}_{shadow_index}.mp4"
            command = [
                args.ffmpeg, "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-i", str(raw), "-t", f"{end - start:.3f}",
                "-map", "0:v:0", "-map", "0:a?",
                "-vf", "scale='min(640,iw)':-2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-y", str(clip),
            ]
            subprocess.run(command, check=True)
            duration = probe_duration(args.ffprobe, clip)
            _duration, frames = extract_frames(
                args.ffmpeg, args.ffprobe, clip, frames_dir, ordinal, args.max_frames
            )
            item = {
                "ordinal": ordinal,
                "uid": record["uid"],
                "title": record.get("title"),
                "shadow_index": shadow_index,
                "demo": demo,
                "source_raw_path": str(raw.relative_to(root)),
                "source_start_sec": start,
                "source_end_sec": end,
                "requested_duration_sec": end - start,
                "rendered_duration_sec": duration,
                "audit_proxy": True,
                "clip_path": str(clip.relative_to(args.out)),
                "clip_sha256": sha256_file(clip),
                "frames": frames,
            }
            item["frame_manifest_sha256"] = sha256_json(frames)
            items.append(item)
            ordinal += 1

    manifest = {
        "kind": f"instructional_{shadow_name}_rendered_outputs",
        "shadow_name": shadow_name,
        "comparison_sha256": hashlib.sha256(args.compare.read_bytes()).hexdigest(),
        "shadow_prompt_sha256": comparison.get("shadow_prompt_sha256"),
        "corpus_mutated": False,
        "items": items,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    with (args.out / "manual_review.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "ordinal", "uid", "shadow_index", "decision", "social_norm",
                "visual_demo", "polarity_supported", "norm_supported",
                "interval_tight", "rejection_reasons", "manual_description",
            ]
        )
        for item in items:
            writer.writerow([item["ordinal"], item["uid"], item["shadow_index"]] + [""] * 8)
    print(json.dumps({"outputs": len(items), "rendered": sum("clip_path" in i for i in items)}))


if __name__ == "__main__":
    main()
