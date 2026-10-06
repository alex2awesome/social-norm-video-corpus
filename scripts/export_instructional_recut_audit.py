#!/usr/bin/env python3
"""Render shadow-only frames for manually proposed instructional recut windows."""

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
from typing import Any

try:
    from .visual_audit_ledger import connect, transform_has_audio_removal, trim_label_transcript
    from .export_visual_audit_batch import extract_frames, probe_duration
except ImportError:  # Direct script execution.
    from visual_audit_ledger import connect, transform_has_audio_removal, trim_label_transcript
    from export_visual_audit_batch import extract_frames, probe_duration


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


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


def extract_window_frames(
    ffmpeg: str,
    clip: Path,
    frames_dir: Path,
    ordinal: int,
    start: float,
    end: float,
    max_frames: int,
) -> list[dict[str, Any]]:
    duration = end - start
    frame_count = min(max_frames, max(4, math.ceil(duration)))
    frames = []
    for index in range(frame_count):
        requested = start + duration * ((index + 0.5) / frame_count)
        target = frames_dir / f"{ordinal:04d}_f{index:02d}.jpg"
        actual = None
        errors = []
        for backoff in (0.0, 0.25, 0.5, 1.0, 2.0):
            candidate = max(start, requested - backoff)
            target.unlink(missing_ok=True)
            result = subprocess.run(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{candidate:.3f}",
                    "-i", str(clip), "-frames:v", "1", "-vf", "scale=768:-2",
                    "-strict", "unofficial", "-threads", "1", "-q:v", "3", "-y", str(target),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and target.is_file() and target.stat().st_size > 0:
                actual = candidate
                break
            errors.append(result.stderr.strip()[-500:])
        if actual is None:
            raise RuntimeError(f"failed frame {ordinal}:{index}: {errors[-1] if errors else ''}")
        frames.append(
            {
                "frame_index": index,
                "recut_timestamp": round(actual - start, 3),
                "source_timestamp": round(actual, 3),
                "requested_source_timestamp": round(requested, 3),
                "path": f"frames/{target.name}",
            }
        )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("windows", type=Path)
    parser.add_argument("--db", type=Path, default=Path("data/visual_audit/audit.db"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--source-clip-field",
        choices=("clip_path", "repaired_clip_path"),
        default="clip_path",
        help="Use repaired_clip_path to chain a recut after a prior rendered repair.",
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    source = json.loads(args.source_manifest.read_text())
    items = {item["item_id"]: item for item in source["items"]}
    windows = json.loads(args.windows.read_text())
    if not isinstance(windows, dict) or not windows:
        raise SystemExit("windows must be a nonempty object keyed by item_id")
    if set(windows) - set(items):
        raise SystemExit(f"unknown items: {sorted(set(windows) - set(items))}")
    args.out.mkdir(parents=True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir()
    clips_dir = args.out / "repaired_clips"
    clips_dir.mkdir()
    project_root = args.project_root.resolve()
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"recut_{stamp}_{hashlib.sha1(args.windows.read_bytes()).hexdigest()[:8]}"
    conn = connect(args.db)
    manifest_path = args.out / "manifest.json"
    conn.execute(
        """
        INSERT INTO batches(
            batch_id,pillar,rubric_version,intended_model,selection_strategy,
            seed,manifest_path,created_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            batch_id, "instructional", "instructional_recut_v1", args.model,
            "manual_quote_local_windows", sha256_json(windows),
            str(manifest_path.resolve()), time.time(),
        ),
    )
    records = []
    for ordinal, item_id in enumerate(windows):
        item = items[item_id]
        spec = windows[item_id]
        start = float(spec["start_sec"])
        end = float(spec["end_sec"])
        source_duration = float(item["duration"])
        if not 0 <= start < end <= source_duration:
            raise SystemExit(f"invalid window for {item_id}: {start}:{end}/{source_duration}")
        if args.source_clip_field == "repaired_clip_path":
            clip = args.source_manifest.resolve().parent / item["repaired_clip_path"]
        else:
            clip = project_root / item["clip_path"]
        repaired_clip = clips_dir / f"{ordinal:04d}.mp4"
        subprocess.run(
            [
                args.ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}",
                "-i", str(clip), "-t", f"{end - start:.3f}", "-map", "0:v:0",
                "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-y",
                str(repaired_clip),
            ],
            check=True,
        )
        repaired_duration = probe_duration(args.ffprobe, repaired_clip)
        if abs(repaired_duration - (end - start)) > 0.35:
            raise RuntimeError(
                f"recut duration mismatch for {item_id}: {repaired_duration:.3f} vs {end-start:.3f}"
            )
        _, frames = extract_frames(
            args.ffmpeg, args.ffprobe, repaired_clip, frames_dir, ordinal, args.max_frames
        )
        source_label_transcript = trim_label_transcript(item, start, end)
        prior_transform = item.get("transform") or {}
        output_audio_removed = transform_has_audio_removal(prior_transform)
        source_audio_stream_count = audio_stream_count(args.ffprobe, clip)
        repaired_audio_stream_count = audio_stream_count(args.ffprobe, repaired_clip)
        if repaired_audio_stream_count != source_audio_stream_count:
            raise RuntimeError(
                f"recut audio stream mismatch for {item_id}: "
                f"{repaired_audio_stream_count} vs {source_audio_stream_count}"
            )
        if output_audio_removed and repaired_audio_stream_count != 0:
            raise RuntimeError(f"audio unexpectedly present after upstream removal: {item_id}")
        record = {
            **{key: value for key, value in item.items() if key not in {"frames", "aligned_transcript"}},
            "ordinal": ordinal,
            "source_batch_id": source["batch_id"],
            "source_frame_manifest_sha256": item["frame_manifest_sha256"],
            "candidate_start_sec": start,
            "candidate_end_sec": end,
            "candidate_duration": end - start,
            "repaired_duration": repaired_duration,
            "window_reason": spec.get("reason"),
            "repair_type": "tighten_to_demo",
            "transform": {
                "prior_repair_type": item.get("repair_type"),
                "prior_transform": prior_transform,
                "output_audio_removed": output_audio_removed,
                "source_audio_stream_count": source_audio_stream_count,
                "repaired_audio_stream_count": repaired_audio_stream_count,
            },
            "repaired_clip_path": str(repaired_clip.relative_to(args.out)),
            "repaired_clip_sha256": sha256_file(repaired_clip),
            "frames": frames,
            "aligned_transcript": [] if output_audio_removed else source_label_transcript,
            "source_label_transcript": source_label_transcript if output_audio_removed else [],
        }
        record["frame_manifest_sha256"] = sha256_json(frames)
        records.append(record)
        conn.execute(
            """
            INSERT INTO batch_items(
                batch_id,item_id,ordinal,frame_count,frame_manifest_sha256
            ) VALUES (?,?,?,?,?)
            """,
            (batch_id, item_id, ordinal, len(frames), record["frame_manifest_sha256"]),
        )
    manifest = {
        "batch_id": batch_id,
        "pillar": "instructional",
        "rubric_version": "instructional_recut_v1",
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
                "ordinal", "item_id", "decision", "candidate_start_sec", "candidate_end_sec",
                "polarity_supported", "norm_supported", "rejection_reasons", "manual_description",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["ordinal"], record["item_id"], "", record["candidate_start_sec"],
                    record["candidate_end_sec"], "", "", "", "",
                ]
            )
    conn.commit()
    conn.close()
    print(json.dumps({"batch_id": batch_id, "items": len(records), "frames": sum(len(r["frames"]) for r in records)}))


if __name__ == "__main__":
    main()
