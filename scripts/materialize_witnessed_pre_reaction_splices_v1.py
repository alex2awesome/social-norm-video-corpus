#!/usr/bin/env python3
"""Materialize exact pre-reaction witnessed candidates with source audio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2

try:
    from scripts.render_dense_blind_followup import make_dense_sheet
    from scripts.score_visual_scene_baselines import sample_frames
except ModuleNotFoundError:
    from render_dense_blind_followup import make_dense_sheet  # type: ignore[no-redef]
    from score_visual_scene_baselines import sample_frames  # type: ignore[no-redef]


REVIEW_FIELDS = (
    "candidate_id", "uid", "social_action_complete",
    "exact_behavior_label_supported", "reaction_visible_absent",
    "reaction_audible_absent", "start_boundary_clean", "end_boundary_clean",
    "artifact_integrity", "manual_description", "manual_rationale",
    "final_manual_accept",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ffmpeg_argv(ffmpeg: str, row: dict[str, Any], target: Path) -> list[str]:
    start, end = float(row["action_start_sec"]), float(row["action_end_sec"])
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(row["source_path"]), "-ss", f"{start:.3f}",
        "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target),
    ]


def probe_media(ffprobe: str, path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries",
            "format=duration:stream=codec_type", "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True, timeout=60,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    try:
        duration = float((payload.get("format") or {})["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unprobeable duration: {path}") from exc
    return {
        "duration_sec": duration,
        "has_video": any(stream.get("codec_type") == "video" for stream in streams),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def write_template(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            value = {field: "" for field in REVIEW_FIELDS}
            value.update({"candidate_id": row["candidate_id"], "uid": row["uid"]})
            writer.writerow(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--storyboard-frames", type=int, default=48)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    reviewed = read_jsonl(args.plan)
    if not reviewed or any(
        row.get("boundary_review_complete") is not True
        or row.get("approval_status") != "unreviewed_exact_splice_plan"
        or row.get("ready_for_splice") not in {True, False}
        for row in reviewed
    ):
        raise ValueError("all splice plans must be fully reviewed and unapproved")
    rows = [row for row in reviewed if row["ready_for_splice"]]
    args.out.mkdir(parents=True)
    clips, sheets = args.out / "clips", args.out / "blind_sheets"
    clips.mkdir(); sheets.mkdir()
    rendered, failures = [], []
    for index, row in enumerate(rows):
        name = hashlib.sha256(row["candidate_id"].encode()).hexdigest()[:20]
        target, sheet = clips / f"{name}.mp4", sheets / f"{name}.jpg"
        try:
            source = Path(row["source_path"])
            if not source.is_file() or sha256(source) != row["source_sha256"]:
                raise ValueError("missing or hash-mismatched source candidate")
            source_probe = probe_media(args.ffprobe, source)
            if not source_probe["has_video"]:
                raise ValueError("source candidate has no video stream")
            subprocess.run(
                ffmpeg_argv(args.ffmpeg, row, target), check=True,
                capture_output=True, text=True, timeout=300,
            )
            artifact_probe = probe_media(args.ffprobe, target)
            expected_duration = float(row["action_end_sec"]) - float(row["action_start_sec"])
            if not artifact_probe["has_video"]:
                raise ValueError("rendered artifact has no video stream")
            if source_probe["has_audio"] and not artifact_probe["has_audio"]:
                raise ValueError("source audio was lost during rendering")
            if abs(artifact_probe["duration_sec"] - expected_duration) > 0.15:
                raise ValueError("rendered duration differs from reviewed boundary by >150ms")
            frames, meta = sample_frames(target, args.storyboard_frames)
            if len(frames) < min(12, args.storyboard_frames):
                raise ValueError("too few frames for splice audit")
            image = make_dense_sheet(frames, meta["sampled_timestamps"], index)
            if not cv2.imwrite(str(sheet), image, [cv2.IMWRITE_JPEG_QUALITY, 93]):
                raise OSError(f"failed to write {sheet}")
            rendered.append({
                **row,
                "artifact_path": str(target.resolve()),
                "artifact_sha256": sha256(target),
                "blind_sheet_path": str(sheet.resolve()),
                "blind_sheet_sha256": sha256(sheet),
                "source_has_audio": source_probe["has_audio"],
                "artifact_has_audio": artifact_probe["has_audio"],
                "audio_preserved": source_probe["has_audio"] == artifact_probe["has_audio"],
                "artifact_duration_sec": artifact_probe["duration_sec"],
                "expected_duration_sec": expected_duration,
                "approval_status": "awaiting_post_splice_manual_audit",
                "automatic_acceptance": False,
                "corpus_mutation_authorized": False,
            })
        except Exception as exc:
            failures.append({**row, "error": f"{type(exc).__name__}: {exc}"})
    manifest = args.out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rendered))
    (args.out / "failures.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures)
    )
    write_template(args.out / "post_splice_manual_review.tsv", rendered)
    (args.out / "summary.json").write_text(json.dumps({
        "kind": "witnessed_pre_reaction_splice_candidates_v1",
        "reviewed": len(reviewed), "eligible": len(rows),
        "rendered": len(rendered), "failed": len(failures),
        "all_source_audio_preserved": all(row["audio_preserved"] for row in rendered),
        "individual_clips_accepted": 0,
        "automatic_acceptance": False, "corpus_mutation_authorized": False,
    }, indent=2, sort_keys=True) + "\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
