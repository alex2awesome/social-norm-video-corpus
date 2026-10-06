#!/usr/bin/env python3
"""Render adaptive edge-text crops for caption-leaking commentary clips.

The input clips already have frozen temporal bounds and no audio.  This shadow
experiment removes only recurrent OCR lines confined to the top/bottom bands,
abstains on central text, and requires the retained image to preserve at least
half of the temporal motion energy.  It never changes source media or metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__:
    from scripts.build_commentary_ocr_line_mask_plan import (
        detect_text_line_boxes,
        recurrent_line_boxes,
    )
    from scripts.build_commentary_ocr_mask_plan import sample_sequential_frames
    from scripts.materialize_commentary_caption_crop_experiment import (
        read_jsonl,
        read_tsv,
        sha256,
    )
    from scripts.materialize_commentary_motion_crop_experiment import derive_parents
    from scripts.materialize_commentary_clip_plan import (
        BLIND_MANUAL_FIELDS,
        MIN_AUDIT_FRAMES,
        POST_REVEAL_FIELDS,
        sample_rendered_frames,
        write_manual_template,
    )
    from scripts.export_commentary_unlabeled_benchmark import render_verified_target
    from scripts.render_full_corpus_score_audit import make_item_sheet
else:
    from build_commentary_ocr_line_mask_plan import (
        detect_text_line_boxes,
        recurrent_line_boxes,
    )
    from build_commentary_ocr_mask_plan import sample_sequential_frames
    from materialize_commentary_caption_crop_experiment import read_jsonl, read_tsv, sha256
    from materialize_commentary_motion_crop_experiment import derive_parents
    from materialize_commentary_clip_plan import (
        BLIND_MANUAL_FIELDS,
        MIN_AUDIT_FRAMES,
        POST_REVEAL_FIELDS,
        sample_rendered_frames,
        write_manual_template,
    )
    from export_commentary_unlabeled_benchmark import render_verified_target
    from render_full_corpus_score_audit import make_item_sheet


EDGE_CENTER_LIMIT = 0.35
MINIMUM_CROP_HEIGHT = 0.45
MINIMUM_MOTION_RETAINED = 0.50
EDGE_PADDING = 0.015


def adaptive_vertical_crop(
    boxes: list[tuple[float, float, float, float]],
    *,
    minimum_height: float = MINIMUM_CROP_HEIGHT,
    padding: float = EDGE_PADDING,
) -> tuple[float, float]:
    """Return normalized y/height, rejecting recurrent text in the middle."""
    if not boxes:
        raise ValueError("no recurrent OCR lines")
    upper: list[tuple[float, float, float, float]] = []
    lower: list[tuple[float, float, float, float]] = []
    central: list[tuple[float, float, float, float]] = []
    for box in boxes:
        _, y, _, height = box
        center = y + height / 2
        if center < EDGE_CENTER_LIMIT:
            upper.append(box)
        elif center > 1 - EDGE_CENTER_LIMIT:
            lower.append(box)
        else:
            central.append(box)
    if central:
        raise ValueError("recurrent OCR line intersects central event region")
    if not upper and not lower:
        raise ValueError("no recurrent edge OCR lines")
    y0 = max((y + height for _, y, _, height in upper), default=0.0) + (
        padding if upper else 0.0
    )
    y1 = min((y for _, y, _, _ in lower), default=1.0) - (
        padding if lower else 0.0
    )
    y0 = min(max(y0, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    if y1 <= y0 or y1 - y0 < minimum_height:
        raise ValueError("safe OCR-free center is too short")
    return y0, y1 - y0


def motion_energy_retained(frames: list[np.ndarray], y: float, height: float) -> float:
    if len(frames) < 2:
        raise ValueError("at least two frames are required")
    shape = frames[0].shape[:2]
    if any(frame.shape[:2] != shape for frame in frames):
        raise ValueError("frames do not share dimensions")
    gray = [
        cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        for frame in frames
    ]
    energy = np.sum([
        cv2.absdiff(left, right).astype(np.float64)
        for left, right in zip(gray, gray[1:])
    ], axis=0)
    total = float(energy.sum())
    if total <= 0:
        raise ValueError("clip has no temporal motion energy")
    frame_height = shape[0]
    top = max(0, min(frame_height - 1, int(round(y * frame_height))))
    bottom = max(top + 1, min(frame_height, int(round((y + height) * frame_height))))
    return float(energy[top:bottom].sum() / total)


def crop_filter(y: float, height: float) -> str:
    if not (0 <= y < 1 and 0 < height <= 1 and y + height <= 1.000001):
        raise ValueError("invalid normalized crop")
    return (
        f"crop=trunc(iw/2)*2:trunc(ih*{height:.6f}/2)*2:"
        f"0:trunc(ih*{y:.6f}/2)*2"
    )


def probe_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"{path}: invalid timing")
    return frames / fps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--post-reveal", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--languages", default="eng+hin+guj")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite shadow directory: {args.out}")
    sealed = read_jsonl(args.sealed)
    source_by_id = {row["candidate_id"]: row for row in sealed}
    parents = derive_parents(sealed, read_tsv(args.blind), read_tsv(args.post_reveal))
    trials: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    for parent in parents:
        source = source_by_id[parent["parent_candidate_id"]]
        proxy = args.proxy_root / f"{int(source['audit_index']):04d}.mp4"
        try:
            frames = sample_sequential_frames(proxy, 8)
            frame_height, frame_width = frames[0].shape[:2]
            lines = [
                detect_text_line_boxes(frame, languages=args.languages)
                for frame in frames
            ]
            boxes = recurrent_line_boxes(
                lines, frame_width, frame_height,
                maximum_box_area=0.20, maximum_total_area=1.0,
            )
            y, height = adaptive_vertical_crop(boxes)
            retained = motion_energy_retained(frames, y, height)
            if retained < MINIMUM_MOTION_RETAINED:
                raise ValueError(
                    f"retained motion {retained:.6f} below {MINIMUM_MOTION_RETAINED:.2f}"
                )
            duration = probe_duration(proxy)
        except (OSError, RuntimeError, ValueError) as exc:
            abstentions.append({
                "parent_candidate_id": parent["parent_candidate_id"],
                "reason": str(exc),
            })
            continue
        trials.append({
            "audit_index": len(trials),
            "candidate_id": f"{parent['parent_candidate_id']}--adaptive_edge_crop_v1",
            "parent_candidate_id": parent["parent_candidate_id"],
            "parent_audit_index": int(source["audit_index"]),
            "variant": "adaptive_edge_crop_v1",
            "crop_y_fraction": y,
            "crop_height_fraction": height,
            "motion_energy_retained": retained,
            "recurrent_ocr_boxes": boxes,
            "source_path": str(proxy.resolve()),
            "source_start_sec": 0.0,
            "source_end_sec": duration,
            "uid": source["uid"],
            "title": source["title"],
            "policy": "shadow_adaptive_edge_crop_only",
        })
    if not trials:
        raise SystemExit("all adaptive edge crops abstained")

    clips = args.out / "clips"
    sheets = args.out / "blind_sheets"
    clips.mkdir(parents=True)
    sheets.mkdir()
    blind_manifest: list[dict[str, Any]] = []
    sealed_manifest: list[dict[str, Any]] = []
    for trial in trials:
        index = int(trial["audit_index"])
        target = clips / f"{index:04d}.mp4"
        seek_mode, timing = render_verified_target(
            Path(trial["source_path"]), target,
            trial["source_start_sec"], trial["source_end_sec"],
            ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, strip_audio=True,
            spatial_filter=crop_filter(
                trial["crop_y_fraction"], trial["crop_height_fraction"]
            ),
        )
        frames, media = sample_rendered_frames(target, timing)
        if len(frames) < MIN_AUDIT_FRAMES:
            raise RuntimeError(f"{trial['candidate_id']}: only {len(frames)} audit frames")
        sheet = make_item_sheet(frames, media["sampled_timestamps"], index)
        sheet_path = sheets / f"{index:04d}.jpg"
        if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {sheet_path}")
        public = {
            "audit_index": index,
            "candidate_id": trial["candidate_id"],
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
        sealed_manifest.append({**public, **trial})

    blind_path = args.out / "blind_manifest.jsonl"
    sealed_path = args.out / "sealed_manifest.jsonl"
    blind_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in blind_manifest))
    sealed_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed_manifest))
    blind_fields = [*BLIND_MANUAL_FIELDS, "editorial_target_marker_absent"]
    write_manual_template(args.out / "blind_manual_review.tsv", blind_manifest, blind_fields)
    write_manual_template(args.out / "post_reveal_review.tsv", blind_manifest, POST_REVEAL_FIELDS)
    summary = {
        "kind": "commentary_adaptive_edge_crop_materialization_v1",
        "cohort_items": len(parents),
        "rendered_items": len(trials),
        "abstention_count": len(abstentions),
        "abstentions": abstentions,
        "minimum_crop_height_fraction": MINIMUM_CROP_HEIGHT,
        "minimum_motion_energy_retained": MINIMUM_MOTION_RETAINED,
        "all_sources_preserved": True,
        "automatic_acceptance": False,
        "corpus_mutated": False,
        "artifact_sha256": {
            "blind_manifest": sha256(blind_path),
            "sealed_manifest": sha256(sealed_path),
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
