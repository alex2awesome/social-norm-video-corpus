#!/usr/bin/env python3
"""Render a frozen commentary localization plan as muted shadow clips.

The sealed manifest retains title and proposal provenance.  The blind manifest
contains only opaque identifiers, transformed media, and hashes.  No source
media, corpus metadata, or labels are changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2

if __package__:
    from scripts.export_commentary_unlabeled_benchmark import render_verified_target
    from scripts.render_full_corpus_score_audit import make_item_sheet
    from scripts.score_visual_scene_baselines import sample_frames
else:
    from export_commentary_unlabeled_benchmark import render_verified_target
    from render_full_corpus_score_audit import make_item_sheet
    from score_visual_scene_baselines import sample_frames


VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov", ".m4v"}
MIN_AUDIT_FRAMES = 8
BLIND_MANUAL_FIELDS = (
    "audit_index",
    "candidate_id",
    "performed_event_visible",
    "actor_target_grounded",
    "start_boundary_clean",
    "end_boundary_clean",
    "label_bearing_text_absent",
    "audio_absent",
    "medium",
    "blind_evidence",
    "blind_note",
)
POST_REVEAL_FIELDS = (
    "audit_index",
    "candidate_id",
    "exact_named_action_visible",
    "label_alignment",
    "route",
    "post_reveal_evidence",
    "post_reveal_note",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_plans(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    uids: set[str] = set()
    for row in sorted(rows, key=lambda value: int(value["audit_index"])):
        candidate_id = str(row.get("candidate_id") or "")
        uid = str(row.get("uid") or "")
        source = Path(str(row.get("source_path") or ""))
        start = float(row.get("proposed_start_sec") or 0)
        end = float(row.get("proposed_end_sec") or 0)
        duration = float(row.get("source_duration_sec") or 0)
        if row.get("approval_status") != "unreviewed_candidate":
            raise ValueError(f"{candidate_id}: plan must be unreviewed")
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError(f"empty or duplicate candidate_id: {candidate_id!r}")
        if not uid or uid in uids:
            raise ValueError(f"empty or duplicate uid: {uid!r}")
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"{candidate_id}: unsupported source extension")
        if not (0 <= start < end <= duration + 0.05):
            raise ValueError(f"{candidate_id}: invalid proposed bounds")
        candidate_ids.add(candidate_id)
        uids.add(uid)
        selected.append(row)
    if not selected:
        raise ValueError("commentary clip plan is empty")
    return selected


def write_manual_template(
    path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "audit_index": row["audit_index"],
                "candidate_id": row["candidate_id"],
            })


def sample_rendered_frames(
    target: Path, timing: dict[str, Any], sampler=sample_frames
):
    """Use one bounded sequential decode instead of unreliable random seeks."""
    return sampler(target, 24, 0.0, float(timing["video_duration_sec"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite shadow directory: {args.out}")
    plans = select_plans(read_jsonl(args.plan))
    missing_sources = [
        str(plan["source_path"])
        for plan in plans
        if not Path(plan["source_path"]).is_file()
        or Path(plan["source_path"]).stat().st_size <= 0
    ]
    if missing_sources:
        raise SystemExit(
            "missing or empty source media before rendering: "
            + ", ".join(missing_sources[:10])
        )
    clips = args.out / "clips"
    sheets = args.out / "blind_sheets"
    clips.mkdir(parents=True)
    sheets.mkdir()
    sealed: list[dict[str, Any]] = []
    blind: list[dict[str, Any]] = []
    for plan in plans:
        audit_index = int(plan["audit_index"])
        candidate_id = str(plan["candidate_id"])
        target = clips / f"{audit_index:04d}.mp4"
        seek_mode, timing = render_verified_target(
            Path(plan["source_path"]), target,
            float(plan["proposed_start_sec"]), float(plan["proposed_end_sec"]),
            ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, strip_audio=True,
        )
        frames, media = sample_rendered_frames(target, timing)
        if len(frames) < MIN_AUDIT_FRAMES:
            raise RuntimeError(f"{candidate_id}: only {len(frames)} audit frames")
        sheet = make_item_sheet(frames, media["sampled_timestamps"], audit_index)
        sheet_path = sheets / f"{audit_index:04d}.jpg"
        if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {sheet_path}")
        blind_row = {
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
            "media_timing": timing,
        }
        blind.append(blind_row)
        sealed.append({
            **blind_row,
            "item_id": plan["item_id"],
            "uid": plan["uid"],
            "title": plan["title"],
            "selection_rule": plan["selection_rule"],
            "proposal_source": plan["proposal_source"],
            "source_path": plan["source_path"],
            "source_event_bounds_sec": [
                float(plan["proposed_start_sec"]),
                float(plan["proposed_end_sec"]),
            ],
            "qwen_bounds_sec": plan.get("qwen_bounds_sec"),
            "gemma_bounds_sec": plan.get("gemma_bounds_sec"),
            "policy": "shadow_transform_manual_audit_only",
        })
    sealed_path = args.out / "sealed_manifest.jsonl"
    blind_path = args.out / "blind_manifest.jsonl"
    sealed_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed))
    blind_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in blind))
    blind_manual_path = args.out / "blind_manual_review.tsv"
    post_reveal_path = args.out / "post_reveal_review.tsv"
    write_manual_template(blind_manual_path, blind, BLIND_MANUAL_FIELDS)
    write_manual_template(post_reveal_path, blind, POST_REVEAL_FIELDS)
    summary = {
        "kind": "commentary_clip_plan_shadow_materialization_v1",
        "items": len(blind),
        "unique_sources": len({row["uid"] for row in sealed}),
        "audio_preserved": False,
        "model_or_semantic_fields_in_blind_manifest": False,
        "source_mutated": False,
        "metadata_mutated": False,
        "automatic_acceptance": False,
        "artifact_sha256": {
            "plan": sha256(args.plan),
            "sealed_manifest": sha256(sealed_path),
            "blind_manifest": sha256(blind_path),
            "blind_manual_template": sha256(blind_manual_path),
            "post_reveal_template": sha256(post_reveal_path),
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
