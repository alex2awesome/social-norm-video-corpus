#!/usr/bin/env python3
"""Render fixed spatial crops for exact commentary scenes with label text.

This is a shadow remediation experiment. It derives its cohort from completed
manual ledgers, applies every preregistered crop to every eligible source, and
never changes source media or corpus metadata.
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
    from materialize_commentary_clip_plan import (
        BLIND_MANUAL_FIELDS,
        MIN_AUDIT_FRAMES,
        POST_REVEAL_FIELDS,
        sample_rendered_frames,
        write_manual_template,
    )
    from render_full_corpus_score_audit import make_item_sheet


CROP_VARIANTS = {
    # Removes typical top station bugs and bottom lower-thirds while retaining
    # the full horizontal field of view.
    "bands_08_20": (
        "crop=trunc(iw/2)*2:trunc(ih*0.72/2)*2:0:trunc(ih*0.08/2)*2"
    ),
    # Tests whether the central incident inset can be isolated from a full news
    # layout. The constants are global and cannot be tuned per source.
    "center_90_64": (
        "crop=trunc(iw*0.90/2)*2:trunc(ih*0.64/2)*2:"
        "trunc(iw*0.05/2)*2:trunc(ih*0.12/2)*2"
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def keyed(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output = {str(row.get("candidate_id") or ""): row for row in rows}
    if "" in output or len(output) != len(rows):
        raise ValueError(f"{name} contains an empty or duplicate candidate_id")
    return output


def derive_trials(
    sealed: list[dict[str, Any]],
    blind: list[dict[str, str]],
    post: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Select all exact visual scenes whose prior transform leaked text."""
    sources = keyed(sealed, "sealed manifest")
    visuals = keyed(blind, "blind ledger")
    semantics = keyed(post, "post-reveal ledger")
    if not (set(sources) == set(visuals) == set(semantics)):
        raise ValueError("input artifacts do not cover the same candidates")
    trials: list[dict[str, Any]] = []
    for parent_id, source in sorted(
        sources.items(), key=lambda value: int(value[1]["audit_index"])
    ):
        visual = visuals[parent_id]
        semantic = semantics[parent_id]
        eligible = (
            visual.get("performed_event_visible") == "yes"
            and visual.get("actor_target_grounded") == "yes"
            and visual.get("label_bearing_text_absent") == "no"
            and semantic.get("exact_named_action_visible") == "yes"
            and semantic.get("label_alignment") == "exact"
            and semantic.get("route") == "commentary_visual"
        )
        if not eligible:
            continue
        start, end = source["source_event_bounds_sec"]
        for variant, spatial_filter in CROP_VARIANTS.items():
            trials.append({
                "audit_index": len(trials),
                "candidate_id": f"{parent_id}--{variant}",
                "parent_candidate_id": parent_id,
                "parent_audit_index": int(source["audit_index"]),
                "variant": variant,
                "spatial_filter": spatial_filter,
                "source_path": source["source_path"],
                "source_start_sec": float(start),
                "source_end_sec": float(end),
                "uid": source["uid"],
                "title": source["title"],
            })
    if not trials:
        raise ValueError("no exact caption-leaking commentary scenes were eligible")
    expected_variants = set(CROP_VARIANTS)
    by_parent: dict[str, set[str]] = {}
    for trial in trials:
        by_parent.setdefault(trial["parent_candidate_id"], set()).add(trial["variant"])
    if any(variants != expected_variants for variants in by_parent.values()):
        raise AssertionError("not every eligible parent received every crop variant")
    return trials


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
    trials = derive_trials(
        read_jsonl(args.sealed), read_tsv(args.blind), read_tsv(args.post_reveal)
    )
    missing = sorted({
        trial["source_path"] for trial in trials
        if not Path(trial["source_path"]).is_file()
        or Path(trial["source_path"]).stat().st_size <= 0
    })
    if missing:
        raise SystemExit("missing or empty source media: " + ", ".join(missing[:10]))
    clips = args.out / "clips"
    sheets = args.out / "blind_sheets"
    clips.mkdir(parents=True)
    sheets.mkdir()
    blind_manifest: list[dict[str, Any]] = []
    sealed_manifest: list[dict[str, Any]] = []
    for trial in trials:
        audit_index = int(trial["audit_index"])
        target = clips / f"{audit_index:04d}.mp4"
        seek_mode, timing = render_verified_target(
            Path(trial["source_path"]), target,
            trial["source_start_sec"], trial["source_end_sec"],
            ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, strip_audio=True,
            spatial_filter=trial["spatial_filter"],
        )
        frames, media = sample_rendered_frames(target, timing)
        if len(frames) < MIN_AUDIT_FRAMES:
            raise RuntimeError(
                f"{trial['candidate_id']}: only {len(frames)} audit frames"
            )
        sheet = make_item_sheet(frames, media["sampled_timestamps"], audit_index)
        sheet_path = sheets / f"{audit_index:04d}.jpg"
        if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {sheet_path}")
        public = {
            "audit_index": audit_index,
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
        sealed_manifest.append({**public, **trial, "policy": "shadow_crop_audit_only"})
    blind_path = args.out / "blind_manifest.jsonl"
    sealed_path = args.out / "sealed_manifest.jsonl"
    blind_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind_manifest)
    )
    sealed_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed_manifest)
    )
    write_manual_template(
        args.out / "blind_manual_review.tsv", blind_manifest, BLIND_MANUAL_FIELDS
    )
    write_manual_template(
        args.out / "post_reveal_review.tsv", blind_manifest, POST_REVEAL_FIELDS
    )
    summary = {
        "kind": "commentary_caption_crop_shadow_materialization_v1",
        "items": len(trials),
        "parents": len({row["parent_candidate_id"] for row in trials}),
        "variants": sorted(CROP_VARIANTS),
        "all_variants_applied_to_every_parent": True,
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
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
