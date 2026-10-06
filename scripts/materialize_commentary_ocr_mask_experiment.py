#!/usr/bin/env python3
"""Render original commentary intervals with preregistered OCR boxes masked."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

if __package__:
    from scripts.export_commentary_unlabeled_benchmark import render_verified_target
    from scripts.materialize_commentary_caption_crop_experiment import read_jsonl, sha256
    from scripts.materialize_commentary_clip_plan import (
        BLIND_MANUAL_FIELDS,
        MIN_AUDIT_FRAMES,
        POST_REVEAL_FIELDS,
        sample_rendered_frames,
        write_manual_template,
    )
    from scripts.render_full_corpus_score_audit import make_item_sheet
else:
    from export_commentary_unlabeled_benchmark import render_verified_target
    from materialize_commentary_caption_crop_experiment import read_jsonl, sha256
    from materialize_commentary_clip_plan import (
        BLIND_MANUAL_FIELDS,
        MIN_AUDIT_FRAMES,
        POST_REVEAL_FIELDS,
        sample_rendered_frames,
        write_manual_template,
    )
    from render_full_corpus_score_audit import make_item_sheet


def validate_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_ids: set[str] = set()
    parents: set[str] = set()
    for row in rows:
        candidate = str(row.get("candidate_id") or "")
        parent = str(row.get("parent_candidate_id") or "")
        boxes = row.get("mask_boxes_normalized")
        if not candidate or candidate in candidate_ids:
            raise ValueError("empty or duplicate candidate_id")
        if not parent or parent in parents:
            raise ValueError("empty or duplicate parent_candidate_id")
        if row.get("variant") not in {"ocr_mask_v1", "ocr_line_mask_v2"}:
            raise ValueError(f"{candidate}: unexpected variant")
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f"{candidate}: missing mask boxes")
        for box in boxes:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"{candidate}: malformed mask box")
            x, y, width, height = map(float, box)
            if not (0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0):
                raise ValueError(f"{candidate}: invalid mask box")
            if x + width > 1.000001 or y + height > 1.000001:
                raise ValueError(f"{candidate}: mask box exceeds frame")
        source = Path(str(row.get("source_path") or ""))
        if source.suffix.lower() not in {".mp4", ".webm", ".mkv", ".mov", ".m4v"}:
            raise ValueError(f"{candidate}: invalid source path")
        if not 0 <= float(row["source_start_sec"]) < float(row["source_end_sec"]):
            raise ValueError(f"{candidate}: invalid temporal bounds")
        candidate_ids.add(candidate)
        parents.add(parent)
    if not rows:
        raise ValueError("OCR mask plan is empty")
    return sorted(rows, key=lambda row: int(row["audit_index"]))


def mask_filter(boxes: list[list[float]]) -> str:
    filters = []
    for x, y, width, height in boxes:
        filters.append(
            "drawbox="
            f"x=iw*{float(x):.6f}:y=ih*{float(y):.6f}:"
            f"w=iw*{float(width):.6f}:h=ih*{float(height):.6f}:"
            "color=black:t=fill"
        )
    if not filters:
        raise ValueError("no OCR boxes to mask")
    return ",".join(filters)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite shadow directory: {args.out}")
    plans = validate_plan(read_jsonl(args.plan))
    if any(not Path(row["source_path"]).is_file() for row in plans):
        raise SystemExit("one or more source videos are missing")
    clips = args.out / "clips"
    sheets = args.out / "blind_sheets"
    clips.mkdir(parents=True)
    sheets.mkdir()
    blind_manifest: list[dict[str, Any]] = []
    sealed_manifest: list[dict[str, Any]] = []
    for plan in plans:
        audit_index = int(plan["audit_index"])
        target = clips / f"{audit_index:04d}.mp4"
        seek_mode, timing = render_verified_target(
            Path(plan["source_path"]), target,
            float(plan["source_start_sec"]), float(plan["source_end_sec"]),
            ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, strip_audio=True,
            spatial_filter=mask_filter(plan["mask_boxes_normalized"]),
        )
        frames, media = sample_rendered_frames(target, timing)
        if len(frames) < MIN_AUDIT_FRAMES:
            raise RuntimeError(f"{plan['candidate_id']}: only {len(frames)} audit frames")
        sheet = make_item_sheet(frames, media["sampled_timestamps"], audit_index)
        sheet_path = sheets / f"{audit_index:04d}.jpg"
        if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {sheet_path}")
        public = {
            "audit_index": audit_index,
            "candidate_id": plan["candidate_id"],
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
        sealed_manifest.append({**public, **plan, "policy": "shadow_ocr_mask_audit_only"})
    blind_path = args.out / "blind_manifest.jsonl"
    sealed_path = args.out / "sealed_manifest.jsonl"
    blind_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in blind_manifest))
    sealed_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed_manifest))
    write_manual_template(args.out / "blind_manual_review.tsv", blind_manifest, BLIND_MANUAL_FIELDS)
    write_manual_template(args.out / "post_reveal_review.tsv", blind_manifest, POST_REVEAL_FIELDS)
    summary = {
        "kind": "commentary_ocr_mask_shadow_materialization_v1",
        "items": len(plans),
        "parents": len(plans),
        "variant": plans[0]["variant"],
        "audio_preserved": False,
        "automatic_acceptance": False,
        "corpus_mutated": False,
        "artifact_sha256": {
            "plan": sha256(args.plan),
            "sealed_manifest": sha256(sealed_path),
            "blind_manifest": sha256(blind_path),
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
