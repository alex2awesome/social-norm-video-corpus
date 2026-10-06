#!/usr/bin/env python3
"""Render dense storyboards around clip-wide witnessed reaction candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2

if __package__:
    from scripts.render_dense_blind_followup import make_dense_sheet
    from scripts.score_visual_scene_baselines import resolve_clip, sample_frames
else:
    from render_dense_blind_followup import make_dense_sheet
    from score_visual_scene_baselines import resolve_clip, sample_frames


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


def candidate_rows(
    proposals: list[dict[str, Any]],
    clips: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    clip_by = {row["item_id"]: row for row in clips}
    rows = []
    for proposal in proposals:
        source = clip_by.get(proposal["item_id"])
        if source is None:
            continue
        for candidate in proposal.get("candidates") or []:
            if candidate.get("negative_self_defense") or candidate.get(
                "negative_reported"
            ):
                continue
            segment_index = int(candidate["segment_index"])
            rows.append(
                {
                    "candidate_id": (
                        f"{proposal['item_id']}:candidate_{segment_index}"
                    ),
                    "item_id": proposal["item_id"],
                    "uid": proposal["uid"],
                    "pillar": "witnessed",
                    "proxy_clip": source["proxy_clip"],
                    "candidate_text": candidate["text"],
                    "candidate_start_sec": float(candidate["start"]),
                    "candidate_end_sec": float(candidate["end"]),
                    "media_start_sec": float(candidate["window_start"]),
                    "media_end_sec": float(candidate["window_end"]),
                    "mechanisms": candidate["mechanisms"],
                }
            )
    return rows


def render(
    row: dict[str, Any],
    index: int,
    clip_manifest: Path,
    output_dir: Path,
    frames: int,
) -> dict[str, Any]:
    clip = resolve_clip(row, clip_manifest)
    sampled, media = sample_frames(
        clip,
        frames,
        row["media_start_sec"],
        row["media_end_sec"],
    )
    if len(sampled) < 4:
        raise RuntimeError(f"{row['candidate_id']}: only {len(sampled)} frames")
    sheet = make_dense_sheet(sampled, media["sampled_timestamps"], index)
    target = output_dir / f"{index:04d}.jpg"
    if not cv2.imwrite(
        str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 91]
    ):
        raise RuntimeError(f"failed to write {target}")
    return {
        **row,
        "storyboard_index": index,
        "sheet_path": str(target),
        "sheet_sha256": sha256(target),
        "frame_count": len(sampled),
        "sampled_timestamps": media["sampled_timestamps"],
        "frame_order": "left_to_right_then_top_to_bottom",
        "corpus_disposition": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--clip-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    if args.out_manifest.exists():
        raise SystemExit(f"refusing to overwrite: {args.out_manifest}")
    if args.frames < 12 or args.frames % 12:
        raise SystemExit("--frames must be a positive multiple of 12")

    rows = candidate_rows(
        load_jsonl(args.proposals),
        load_jsonl(args.clip_manifest),
    )
    if not rows:
        raise SystemExit("no candidate rows could be joined to clips")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rendered = [
        render(row, index, args.clip_manifest, args.out_dir, args.frames)
        for index, row in enumerate(rows)
    ]
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_manifest.with_suffix(".tmp.jsonl")
    with temporary.open("w") as handle:
        for row in rendered:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, args.out_manifest)
    print(
        json.dumps(
            {
                "kind": "witnessed_reaction_candidate_storyboards_v1",
                "candidates": len(rendered),
                "source_proposals_sha256": sha256(args.proposals),
                "manifest_sha256": sha256(args.out_manifest),
                "corpus_mutated": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
