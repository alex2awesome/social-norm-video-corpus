#!/usr/bin/env python3
"""Materialize manually bounded title-event clips as muted shadow artifacts.

Only dense-review rows marked ``strict_recovered`` are materialized. Sources
and metadata are never modified. Audio is always removed so title/narrator
language cannot leak the weak label into the visual training artifact.
"""

from __future__ import annotations

import argparse
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_recovered(
    candidates: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidate_by_item = {row["item_id"]: row for row in candidates}
    if len(candidate_by_item) != len(candidates):
        raise ValueError("duplicate candidate item_id")
    selected = []
    for dense in sorted(dense_rows, key=lambda row: int(row["audit_index"])):
        if dense.get("dense_outcome") != "strict_recovered":
            continue
        start = dense.get("event_start_sec")
        end = dense.get("event_end_sec")
        if start is None or end is None or not 0 <= float(start) < float(end):
            raise ValueError(f"invalid event bounds at audit_index {dense['audit_index']}")
        item_id = dense["item_id"]
        candidate = candidate_by_item.get(item_id)
        if candidate is None:
            raise ValueError(f"recovered item absent from candidates: {item_id}")
        source = Path(candidate["source_path"])
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"unsupported source extension: {source}")
        selected.append((candidate, dense))
    if not selected:
        raise ValueError("no strict recovered rows")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--dense-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite shadow directory: {args.out}")
    selected = select_recovered(
        load_jsonl(args.candidates),
        load_jsonl(args.dense_review),
    )

    args.out.mkdir(parents=True)
    clips = args.out / "clips"
    sheets = args.out / "blind_sheets"
    clips.mkdir()
    sheets.mkdir()
    output = []
    for candidate, dense in selected:
        audit_index = int(dense["audit_index"])
        uid = candidate["uid"]
        source = Path(candidate["source_path"])
        target = clips / f"{audit_index:02d}__{uid}.mp4"
        seek_mode, timing = render_verified_target(
            source,
            target,
            float(dense["event_start_sec"]),
            float(dense["event_end_sec"]),
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            strip_audio=True,
        )
        frames, media = sample_frames(target, 12)
        if len(frames) < 8:
            raise RuntimeError(f"{target} produced only {len(frames)} audit frames")
        sheet = make_item_sheet(
            frames,
            media["sampled_timestamps"],
            audit_index,
        )
        sheet_path = sheets / f"{audit_index:02d}.jpg"
        if not cv2.imwrite(
            str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]
        ):
            raise RuntimeError(f"failed to write {sheet_path}")
        output.append(
            {
                "audit_index": audit_index,
                "item_id": candidate["item_id"],
                "uid": uid,
                "pillar": "commentary_title_event_shadow",
                "proxy_clip": str(target),
                "proxy_clip_sha256": sha256(target),
                "sheet_path": str(sheet_path),
                "sheet_sha256": sha256(sheet_path),
                "source_event_bounds_sec": [
                    float(dense["event_start_sec"]),
                    float(dense["event_end_sec"]),
                ],
                "transform": "temporal_cut_and_mute",
                "audio_preserved": False,
                "seek_mode": seek_mode,
                "media_timing": timing,
            }
        )

    manifest = args.out / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output)
    )
    summary = {
        "kind": "commentary_title_event_shadow_clips",
        "items": len(output),
        "audio_preserved": False,
        "source_mutated": False,
        "metadata_mutated": False,
        "manifest_sha256": sha256(manifest),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
