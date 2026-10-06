#!/usr/bin/env python3
"""Render a fixed last-half trim for caption-leaking commentary scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

if __package__:
    from scripts.export_commentary_unlabeled_benchmark import render_verified_target
    from scripts.materialize_commentary_caption_crop_experiment import (
        derive_trials, read_jsonl, read_tsv, sha256,
    )
    from scripts.materialize_commentary_clip_plan import (
        BLIND_MANUAL_FIELDS, MIN_AUDIT_FRAMES, POST_REVEAL_FIELDS,
        sample_rendered_frames, write_manual_template,
    )
    from scripts.render_full_corpus_score_audit import make_item_sheet
else:
    from export_commentary_unlabeled_benchmark import render_verified_target
    from materialize_commentary_caption_crop_experiment import derive_trials, read_jsonl, read_tsv, sha256
    from materialize_commentary_clip_plan import BLIND_MANUAL_FIELDS, MIN_AUDIT_FRAMES, POST_REVEAL_FIELDS, sample_rendered_frames, write_manual_template
    from render_full_corpus_score_audit import make_item_sheet


def media_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"cannot determine duration: {path}")
    return frames / fps


def tail_bounds(duration: float) -> tuple[float, float]:
    if duration < 2.0:
        raise ValueError("proxy is too short for fixed tail trim")
    return duration * 0.5, duration


def unique_eligible_parents(
    sealed: list[dict[str, Any]],
    blind: list[dict[str, str]],
    post: list[dict[str, str]],
) -> list[dict[str, Any]]:
    # Reuse the audited eligibility contract from the spatial experiment, then
    # collapse its two variants back to one row per parent.
    trials = derive_trials(sealed, blind, post)
    parents: dict[str, dict[str, Any]] = {}
    for trial in trials:
        parents.setdefault(trial["parent_candidate_id"], trial)
    return sorted(parents.values(), key=lambda row: row["parent_audit_index"])


def run(
    sealed_path: Path,
    blind_path: Path,
    post_path: Path,
    proxy_root: Path,
    output_dir: Path,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite: {output_dir}")
    parents = unique_eligible_parents(
        read_jsonl(sealed_path), read_tsv(blind_path), read_tsv(post_path)
    )
    clips = output_dir / "clips"
    sheets = output_dir / "blind_sheets"
    clips.mkdir(parents=True)
    sheets.mkdir()
    blind_manifest = []
    sealed_manifest = []
    for audit_index, parent in enumerate(parents):
        source = proxy_root / f"{int(parent['parent_audit_index']):04d}.mp4"
        if not source.is_file():
            raise ValueError(f"missing local exact-event proxy: {source}")
        duration = media_duration(source)
        start, end = tail_bounds(duration)
        target = clips / f"{audit_index:04d}.mp4"
        seek_mode, timing = render_verified_target(
            source, target, start, end,
            ffmpeg=ffmpeg, ffprobe=ffprobe, strip_audio=True,
        )
        frames, media = sample_rendered_frames(target, timing)
        if len(frames) < MIN_AUDIT_FRAMES:
            raise RuntimeError(f"{parent['parent_candidate_id']}: insufficient frames")
        sheet = make_item_sheet(frames, media["sampled_timestamps"], audit_index)
        sheet_path = sheets / f"{audit_index:04d}.jpg"
        if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {sheet_path}")
        public = {
            "audit_index": audit_index,
            "candidate_id": f"{parent['parent_candidate_id']}--last_half_v1",
            "proxy_clip": str(target.resolve()),
            "proxy_clip_sha256": sha256(target),
            "sheet_path": str(sheet_path.resolve()),
            "sheet_sha256": sha256(sheet_path),
            "frame_count": len(frames),
            "sampled_timestamps": media["sampled_timestamps"],
            "audio_preserved": False,
            "seek_mode": seek_mode,
            "media_timing": timing,
        }
        blind_manifest.append(public)
        sealed_manifest.append({
            **public,
            "parent_candidate_id": parent["parent_candidate_id"],
            "parent_audit_index": parent["parent_audit_index"],
            "uid": parent["uid"],
            "title": parent["title"],
            "variant": "last_half_v1",
            "source_proxy": str(source.resolve()),
            "source_proxy_sha256": sha256(source),
            "relative_start_sec": start,
            "relative_end_sec": end,
            "policy": "shadow_temporal_trim_audit_only",
        })
    blind_manifest_path = output_dir / "blind_manifest.jsonl"
    sealed_manifest_path = output_dir / "sealed_manifest.jsonl"
    blind_manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in blind_manifest))
    sealed_manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed_manifest))
    write_manual_template(output_dir / "blind_manual_review.tsv", blind_manifest, BLIND_MANUAL_FIELDS)
    write_manual_template(output_dir / "post_reveal_review.tsv", blind_manifest, POST_REVEAL_FIELDS)
    summary = {
        "kind": "commentary_fixed_tail_trim_materialization_v1",
        "items": len(parents),
        "parents": len(parents),
        "variant": "last_half_v1",
        "per_video_tuning": False,
        "automatic_acceptance": False,
        "corpus_mutated": False,
        "artifact_sha256": {
            "sealed_input": sha256(sealed_path),
            "blind_input": sha256(blind_path),
            "post_input": sha256(post_path),
            "blind_manifest": sha256(blind_manifest_path),
            "sealed_manifest": sha256(sealed_manifest_path),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--post-reveal", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.sealed, args.blind, args.post_reveal, args.proxy_root, args.out, args.ffmpeg, args.ffprobe), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
