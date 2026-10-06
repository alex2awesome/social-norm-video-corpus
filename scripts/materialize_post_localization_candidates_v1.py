#!/usr/bin/env python3
"""Render exact-cut candidates under a manually sealed audio policy.

Only plans explicitly marked ready may be rendered.  Outputs remain unapproved
until the separate post-transform ledger passes every visual, boundary,
audio/leakage, and integrity field. Commentary is muted; clean instructional
in-character audio may be preserved when the behavior depends on it.
"""

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
    "candidate_id", "pillar", "uid", "visual_event_complete",
    "actor_target_grounded", "exact_behavior_label_supported",
    "start_boundary_clean", "end_boundary_clean",
    "label_bearing_visible_text_absent",
    "audio_policy_correct", "label_bearing_audio_absent",
    "required_behavior_audio_preserved",
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


def validate_plan(rows: list[dict[str, Any]]) -> None:
    ids = [str(row.get("candidate_id") or "") for row in rows]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("artifact plan has missing or duplicate candidate ids")
    for row in rows:
        candidate_id = row["candidate_id"]
        if row.get("ready_for_final_render") is not True:
            raise ValueError(f"{candidate_id}: plan is not ready for final render")
        if row.get("approval_status") != "unreviewed_localization_candidate":
            raise ValueError(f"{candidate_id}: plan is incorrectly pre-approved")
        if row.get("transform_required") not in {
            "temporal_cut_and_mute", "temporal_cut_preserve_audio",
        }:
            raise ValueError(f"{candidate_id}: unsupported or incomplete transform")
        if row.get("pillar") not in {"instructional", "commentary"}:
            raise ValueError(f"{candidate_id}: unsupported pillar")
        if (
            row.get("pillar") == "instructional"
            and row.get("exact_boundary_and_audio_policy_review_complete") is not True
        ):
            raise ValueError(f"{candidate_id}: instructional audio policy is unreviewed")
        if (
            row.get("source_lineage_sealed") is not True
            or not isinstance(row.get("source_sha256"), str)
            or len(row["source_sha256"]) != 64
        ):
            raise ValueError(f"{candidate_id}: source lineage is not hash-sealed")
        start, end = float(row["proposed_start_sec"]), float(row["proposed_end_sec"])
        if not 0 <= start < end:
            raise ValueError(f"{candidate_id}: invalid temporal bounds")
        if not str(row.get("behavior_label") or "").strip():
            raise ValueError(f"{candidate_id}: missing behavior label")


def ready_plans(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = [str(row.get("candidate_id") or "") for row in rows]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("artifact plan has missing or duplicate candidate ids")
    for row in rows:
        if row.get("ready_for_final_render") not in {True, False}:
            raise ValueError(f"{row['candidate_id']}: missing reviewed render disposition")
        if row.get("approval_status") != "unreviewed_localization_candidate":
            raise ValueError(f"{row['candidate_id']}: plan is incorrectly pre-approved")
    eligible = [row for row in rows if row["ready_for_final_render"]]
    if eligible:
        validate_plan(eligible)
    return eligible


def ffmpeg_argv(ffmpeg: str, row: dict[str, Any], target: Path) -> list[str]:
    start, end = float(row["proposed_start_sec"]), float(row["proposed_end_sec"])
    argv = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(row["source_path"]), "-ss", f"{start:.3f}",
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
    ]
    if row["transform_required"] == "temporal_cut_and_mute":
        argv.append("-an")
    else:
        argv.extend(["-map", "0:v:0", "-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k"])
    argv.extend(["-movflags", "+faststart", str(target)])
    return argv


def probe(ffprobe: str, path: Path) -> dict[str, Any]:
    payload = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries",
        "format=duration:stream=codec_type", "-of", "json", str(path),
    ], text=True))
    streams = payload.get("streams") or []
    return {
        "duration_sec": float((payload.get("format") or {}).get("duration") or 0),
        "has_video": any(row.get("codec_type") == "video" for row in streams),
        "has_audio": any(row.get("codec_type") == "audio" for row in streams),
    }


def safe_name(candidate_id: str) -> str:
    return hashlib.sha256(candidate_id.encode()).hexdigest()[:20]


def write_review_template(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            value = {field: "" for field in REVIEW_FIELDS}
            value.update({
                "candidate_id": row["candidate_id"],
                "pillar": row["pillar"],
                "uid": row["uid"],
            })
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
    rows = ready_plans(reviewed)
    args.out.mkdir(parents=True)
    clips, sheets = args.out / "clips", args.out / "blind_sheets"
    clips.mkdir()
    sheets.mkdir()
    rendered, failures = [], []
    for index, row in enumerate(rows):
        source = Path(row["source_path"])
        name = safe_name(row["candidate_id"])
        target, sheet = clips / f"{name}.mp4", sheets / f"{name}.jpg"
        try:
            if not source.is_file() or source.stat().st_size <= 0:
                raise FileNotFoundError(source)
            source_hash = sha256(source)
            expected_hash = row["source_sha256"]
            if source_hash != expected_hash:
                raise ValueError("source hash differs from sealed plan")
            source_media = probe(args.ffprobe, source)
            if not source_media["has_video"]:
                raise ValueError("source has no video stream")
            subprocess.run(
                ffmpeg_argv(args.ffmpeg, row, target), check=True,
                capture_output=True, text=True, timeout=300,
            )
            media = probe(args.ffprobe, target)
            preserve_audio = row["transform_required"] == "temporal_cut_preserve_audio"
            expected_audio = preserve_audio and source_media["has_audio"]
            if (
                not media["has_video"]
                or media["has_audio"] != expected_audio
                or media["duration_sec"] <= 0
            ):
                raise ValueError(f"invalid artifact streams for audio policy: {media}")
            expected_duration = float(row["proposed_end_sec"]) - float(row["proposed_start_sec"])
            if abs(media["duration_sec"] - expected_duration) > 0.15:
                raise ValueError("artifact duration differs from sealed bounds by >150ms")
            frames, frame_meta = sample_frames(target, args.storyboard_frames)
            if len(frames) < min(12, args.storyboard_frames):
                raise ValueError("too few frames for final artifact audit")
            image = make_dense_sheet(frames, frame_meta["sampled_timestamps"], index)
            if not cv2.imwrite(str(sheet), image, [cv2.IMWRITE_JPEG_QUALITY, 93]):
                raise OSError(f"failed to write {sheet}")
            rendered.append({
                **row,
                "source_sha256_observed": source_hash,
                "artifact_path": str(target.resolve()),
                "artifact_sha256": sha256(target),
                "blind_sheet_path": str(sheet.resolve()),
                "blind_sheet_sha256": sha256(sheet),
                "media": media,
                "source_media": source_media,
                "audio_policy": "preserve" if preserve_audio else "mute",
                "audio_preserved": expected_audio,
                "approval_status": "awaiting_post_transform_manual_audit",
                "automatic_acceptance": False,
                "corpus_mutation_authorized": False,
            })
        except Exception as exc:
            failures.append({
                **row,
                "artifact_path": str(target) if target.exists() else None,
                "error": f"{type(exc).__name__}: {exc}",
                "approval_status": "render_failed",
            })
    manifest, failed = args.out / "manifest.jsonl", args.out / "failures.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rendered))
    failed.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in failures))
    write_review_template(args.out / "post_transform_manual_review.tsv", rendered)
    summary = {
        "kind": "post_localization_audio_policy_artifact_candidates_v1",
        "reviewed": len(reviewed), "eligible": len(rows),
        "rendered": len(rendered), "failed": len(failures),
        "audio_policies": {
            policy: sum(row.get("audio_policy") == policy for row in rendered)
            for policy in ("preserve", "mute")
        },
        "individual_clips_accepted": 0,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
        "plan_sha256": sha256(args.plan),
        "manifest_sha256": sha256(manifest),
        "failure_sha256": sha256(failed),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
