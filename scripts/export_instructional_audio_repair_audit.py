#!/usr/bin/env python3
"""Render and register shadow-only instructional audio-removal repairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .export_visual_audit_batch import extract_frames, sha256_json
    from .visual_audit_ledger import connect
except ImportError:  # Direct script execution.
    from export_visual_audit_batch import extract_frames, sha256_json
    from visual_audit_ledger import connect


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source_clip(
    source_manifest: Path,
    item: dict,
    source_clip_field: str,
    project_root: Path,
) -> Path:
    if source_clip_field == "repaired_clip_path":
        return source_manifest.resolve().parent / item["repaired_clip_path"]
    return project_root / item["clip_path"]


def audio_stream_count(ffprobe: str, clip: Path) -> int:
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(clip),
        ],
        capture_output=True, text=True, check=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--item-id", action="append", required=True)
    parser.add_argument("--db", type=Path, default=Path("data/visual_audit/audit.db"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--source-clip-field",
        choices=("clip_path", "repaired_clip_path"),
        default="clip_path",
        help="Use repaired_clip_path when removing audio from an already rendered repair.",
    )
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
    seed = sha256_json(args.item_id)
    batch_id = f"audio_repair_{stamp}_{hashlib.sha1(seed.encode()).hexdigest()[:8]}"
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
            "manual_strip_explanatory_audio", seed, str(manifest_path.resolve()), time.time(),
        ),
    )
    records = []
    for ordinal, item_id in enumerate(args.item_id):
        item = source_items[item_id]
        source_clip = resolve_source_clip(
            args.source_manifest, item, args.source_clip_field, project_root
        )
        repaired_clip = clips_dir / f"{ordinal:04d}.mp4"
        source_audio = audio_stream_count(args.ffprobe, source_clip)
        if source_audio < 1:
            raise SystemExit(f"source has no audio stream: {item_id}")
        subprocess.run(
            [
                args.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source_clip),
                "-map", "0:v:0", "-c:v", "copy", "-an", "-movflags", "+faststart",
                "-y", str(repaired_clip),
            ],
            check=True,
        )
        repaired_audio = audio_stream_count(args.ffprobe, repaired_clip)
        if repaired_audio != 0:
            raise RuntimeError(f"audio remains after repair: {item_id}")
        duration, frames = extract_frames(
            args.ffmpeg, args.ffprobe, repaired_clip, frames_dir, ordinal, args.max_frames
        )
        label_transcript = item.get("aligned_transcript") or item.get("source_label_transcript") or []
        record = {
            **{key: value for key, value in item.items() if key != "frames"},
            "ordinal": ordinal,
            "source_batch_id": source["batch_id"],
            "source_frame_manifest_sha256": item["frame_manifest_sha256"],
            "repair_type": "strip_explanatory_audio",
            "transform": {
                "prior_repair_type": item.get("repair_type"),
                "prior_transform": item.get("transform") or {},
                "audio_removed": True,
                "source_audio_stream_count": source_audio,
                "repaired_audio_stream_count": repaired_audio,
            },
            "repaired_clip_path": str(repaired_clip.relative_to(args.out)),
            "repaired_clip_sha256": sha256_file(repaired_clip),
            "duration": duration,
            "frames": frames,
            "aligned_transcript": [],
            "source_label_transcript": label_transcript,
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
