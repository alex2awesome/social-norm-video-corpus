#!/usr/bin/env python3
"""Render deterministic motion-saliency crops for caption-leaking scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__:
    from scripts.export_commentary_unlabeled_benchmark import render_verified_target
    from scripts.materialize_commentary_caption_crop_experiment import (
        derive_trials as derive_fixed_trials,
        read_jsonl,
        read_tsv,
        sha256,
    )
    from scripts.materialize_commentary_clip_plan import (
        BLIND_MANUAL_FIELDS,
        MIN_AUDIT_FRAMES,
        POST_REVEAL_FIELDS,
        sample_rendered_frames,
        write_manual_template,
    )
    from scripts.render_full_corpus_score_audit import make_item_sheet
    from scripts.score_visual_scene_baselines import sample_frames
else:
    from export_commentary_unlabeled_benchmark import render_verified_target
    from materialize_commentary_caption_crop_experiment import (
        derive_trials as derive_fixed_trials,
        read_jsonl,
        read_tsv,
        sha256,
    )
    from materialize_commentary_clip_plan import (
        BLIND_MANUAL_FIELDS,
        MIN_AUDIT_FRAMES,
        POST_REVEAL_FIELDS,
        sample_rendered_frames,
        write_manual_template,
    )
    from render_full_corpus_score_audit import make_item_sheet
    from score_visual_scene_baselines import sample_frames


def weighted_quantile_index(weights: np.ndarray, quantile: float) -> int:
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("motion weights are empty")
    target = total * quantile
    return int(np.searchsorted(np.cumsum(weights), target, side="left"))


def expanded_interval(low: int, high: int, limit: int, minimum_fraction: float) -> tuple[int, int]:
    minimum = max(2, int(round(limit * minimum_fraction)))
    center = (low + high) / 2
    if high - low < minimum:
        low = int(round(center - minimum / 2))
        high = low + minimum
    if low < 0:
        high -= low
        low = 0
    if high > limit:
        low -= high - limit
        high = limit
    return max(0, low), min(limit, high)


def estimate_motion_bbox(frames: list[np.ndarray]) -> tuple[float, float, float, float]:
    """Return x/y/width/height fractions from temporally changing pixels."""
    if len(frames) < 8:
        raise ValueError("at least eight frames are required for motion cropping")
    height, width = frames[0].shape[:2]
    if height < 32 or width < 32:
        raise ValueError("frames are too small")
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise ValueError("frames do not share dimensions")
    gray = [
        cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        for frame in frames
    ]
    diffs = np.stack([
        cv2.absdiff(previous, current)
        for previous, current in zip(gray, gray[1:])
    ]).astype(np.float32)
    motion = np.percentile(diffs, 75, axis=0)
    positive = motion[motion >= 3]
    if positive.size < height * width * 0.001:
        raise ValueError("insufficient temporal motion")
    threshold = max(6.0, float(np.percentile(positive, 70)))
    mask = (motion >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    retained = np.zeros_like(mask)
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        if area < max(12, height * width * 0.0004):
            continue
        edge_band = y < height * 0.18 or y + component_height > height * 0.82
        ticker_like = component_width / max(component_height, 1) > 5
        if edge_band and ticker_like:
            continue
        retained[labels == label] = 1
    energy = motion * retained
    if np.count_nonzero(energy) < height * width * 0.0005:
        raise ValueError("no non-ticker motion region")
    x_weights = energy.sum(axis=0)
    y_weights = energy.sum(axis=1)
    x0 = weighted_quantile_index(x_weights, 0.01)
    x1 = weighted_quantile_index(x_weights, 0.99) + 1
    y0 = weighted_quantile_index(y_weights, 0.01)
    y1 = weighted_quantile_index(y_weights, 0.99) + 1
    x_pad = int(round(width * 0.06))
    y_pad = int(round(height * 0.06))
    x0, x1 = expanded_interval(x0 - x_pad, x1 + x_pad, width, 0.40)
    y0, y1 = expanded_interval(y0 - y_pad, y1 + y_pad, height, 0.40)
    return x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height


def crop_filter(bbox: tuple[float, float, float, float]) -> str:
    x, y, width, height = bbox
    if not (0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0):
        raise ValueError("invalid normalized motion box")
    if x + width > 1.000001 or y + height > 1.000001:
        raise ValueError("motion box exceeds frame")
    return (
        f"crop=trunc(iw*{width:.6f}/2)*2:trunc(ih*{height:.6f}/2)*2:"
        f"trunc(iw*{x:.6f}/2)*2:trunc(ih*{y:.6f}/2)*2"
    )


def derive_parents(
    sealed: list[dict[str, Any]], blind: list[dict[str, str]],
    post: list[dict[str, str]],
) -> list[dict[str, Any]]:
    fixed = derive_fixed_trials(sealed, blind, post)
    parents: dict[str, dict[str, Any]] = {}
    for trial in fixed:
        parents.setdefault(trial["parent_candidate_id"], trial)
    return sorted(parents.values(), key=lambda row: row["parent_audit_index"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--post-reveal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite shadow directory: {args.out}")
    parents = derive_parents(
        read_jsonl(args.sealed), read_tsv(args.blind), read_tsv(args.post_reveal)
    )
    if any(not Path(row["source_path"]).is_file() for row in parents):
        raise SystemExit("one or more source videos are missing")
    clips = args.out / "clips"
    sheets = args.out / "blind_sheets"
    clips.mkdir(parents=True)
    sheets.mkdir()
    blind_manifest: list[dict[str, Any]] = []
    sealed_manifest: list[dict[str, Any]] = []
    for audit_index, parent in enumerate(parents):
        # The prior proxy is a verified full-frame rendition of the same source
        # interval and is used only to estimate normalized motion coordinates.
        proxy = next(
            row for row in read_jsonl(args.sealed)
            if row["candidate_id"] == parent["parent_candidate_id"]
        )
        timing = proxy["media_timing"]
        motion_frames, _ = sample_frames(
            Path(proxy["proxy_clip"]), 24, 0.0, float(timing["video_duration_sec"])
        )
        bbox = estimate_motion_bbox(motion_frames)
        spatial_filter = crop_filter(bbox)
        candidate_id = f"{parent['parent_candidate_id']}--motion_bbox_v1"
        target = clips / f"{audit_index:04d}.mp4"
        seek_mode, target_timing = render_verified_target(
            Path(parent["source_path"]), target,
            parent["source_start_sec"], parent["source_end_sec"],
            ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, strip_audio=True,
            spatial_filter=spatial_filter,
        )
        frames, media = sample_rendered_frames(target, target_timing)
        if len(frames) < MIN_AUDIT_FRAMES:
            raise RuntimeError(f"{candidate_id}: only {len(frames)} audit frames")
        sheet = make_item_sheet(frames, media["sampled_timestamps"], audit_index)
        sheet_path = sheets / f"{audit_index:04d}.jpg"
        if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {sheet_path}")
        public = {
            "audit_index": audit_index,
            "candidate_id": candidate_id,
            "proxy_clip": str(target.resolve()),
            "proxy_clip_sha256": sha256(target),
            "sheet_path": str(sheet_path.resolve()),
            "sheet_sha256": sha256(sheet_path),
            "frame_count": len(frames),
            "sampled_timestamps": media["sampled_timestamps"],
            "audio_preserved": False,
            "seek_mode": seek_mode,
            "media_timing": target_timing,
        }
        blind_manifest.append(public)
        sealed_manifest.append({
            **public,
            "parent_candidate_id": parent["parent_candidate_id"],
            "parent_audit_index": parent["parent_audit_index"],
            "variant": "motion_bbox_v1",
            "motion_bbox_normalized": bbox,
            "spatial_filter": spatial_filter,
            "source_path": parent["source_path"],
            "source_start_sec": parent["source_start_sec"],
            "source_end_sec": parent["source_end_sec"],
            "uid": parent["uid"],
            "title": parent["title"],
            "policy": "shadow_motion_crop_audit_only",
        })
    blind_path = args.out / "blind_manifest.jsonl"
    sealed_path = args.out / "sealed_manifest.jsonl"
    blind_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in blind_manifest))
    sealed_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed_manifest))
    write_manual_template(args.out / "blind_manual_review.tsv", blind_manifest, BLIND_MANUAL_FIELDS)
    write_manual_template(args.out / "post_reveal_review.tsv", blind_manifest, POST_REVEAL_FIELDS)
    summary = {
        "kind": "commentary_motion_crop_shadow_materialization_v1",
        "items": len(parents),
        "parents": len(parents),
        "variant": "motion_bbox_v1",
        "audio_preserved": False,
        "automatic_acceptance": False,
        "corpus_mutated": False,
        "artifact_sha256": {
            "sealed_input": sha256(args.sealed),
            "blind_input": sha256(args.blind),
            "post_reveal_input": sha256(args.post_reveal),
            "sealed_manifest": sha256(sealed_path),
            "blind_manifest": sha256(blind_path),
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
