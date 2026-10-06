#!/usr/bin/env python3
"""Render muted candidate clips and blind sheets without approving them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

if __package__:
    from scripts.export_commentary_unlabeled_benchmark import render_verified_target
    from scripts.render_full_corpus_score_audit import make_item_sheet
    from scripts.score_visual_scene_baselines import sample_frames
else:
    from export_commentary_unlabeled_benchmark import render_verified_target
    from render_full_corpus_score_audit import make_item_sheet
    from score_visual_scene_baselines import sample_frames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_plan(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty candidate plan")
    seen = set()
    for row in rows:
        index = int(row["audit_index"])
        if index in seen:
            raise ValueError(f"duplicate audit_index: {index}")
        seen.add(index)
        if row.get("approval_status") != "unreviewed_candidate":
            raise ValueError(f"candidate is incorrectly pre-approved: {index}")
        start = float(row["proposed_start_sec"])
        end = float(row["proposed_end_sec"])
        duration = float(row["source_duration_sec"])
        if not 0 <= start < end <= duration + 0.01:
            raise ValueError(f"invalid bounds: {index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite candidate directory: {args.out}")
    rows = load_jsonl(args.plan)
    validate_plan(rows)

    args.out.mkdir(parents=True)
    clips = args.out / "clips"
    sheets = args.out / "blind_sheets"
    clips.mkdir()
    sheets.mkdir()
    manifest_rows = []
    failures = []
    for row in rows:
        index = int(row["audit_index"])
        source = Path(row["source_path"])
        target = clips / f"{index:03d}__{row['uid']}.mp4"
        try:
            seek_mode, timing = render_verified_target(
                source,
                target,
                float(row["proposed_start_sec"]),
                float(row["proposed_end_sec"]),
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                strip_audio=True,
            )
            frames, media = sample_frames(target, 12)
            if len(frames) < 8:
                raise RuntimeError(
                    f"candidate produced only {len(frames)} audit frames"
                )
        except Exception as exc:
            failures.append(
                {
                    **row,
                    "proxy_clip": str(target) if target.exists() else None,
                    "proxy_clip_sha256": sha256(target) if target.exists() else None,
                    "approval_status": "render_failed",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        sheet = make_item_sheet(frames, media["sampled_timestamps"], index)
        sheet_path = sheets / f"{index:03d}.jpg"
        if not cv2.imwrite(
            str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]
        ):
            raise RuntimeError(f"failed to write {sheet_path}")
        manifest_rows.append(
            {
                **row,
                "proxy_clip": str(target),
                "proxy_clip_sha256": sha256(target),
                "sheet_path": str(sheet_path),
                "sheet_sha256": sha256(sheet_path),
                "transform": "candidate_temporal_cut_and_mute",
                "audio_preserved": False,
                "seek_mode": seek_mode,
                "media_timing": timing,
                "approval_status": "awaiting_post_transform_audit",
            }
        )
    manifest = args.out / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows)
    )
    failure_path = args.out / "failures.jsonl"
    failure_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures)
    )
    summary = {
        "kind": "commentary_title_event_candidate_clips",
        "planned_items": len(rows),
        "rendered_items": len(manifest_rows),
        "failed_items": len(failures),
        "audio_preserved": False,
        "approval_status": "awaiting_post_transform_audit",
        "source_mutated": False,
        "metadata_mutated": False,
        "manifest_sha256": sha256(manifest),
        "failures_sha256": sha256(failure_path),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
