#!/usr/bin/env python3
"""Render metadata-blind dense sheets for unresolved title-event candidates.

The post-reveal ledger is used only to select unresolved item IDs. Titles,
norms, cue values, and dispositions are not emitted into the dense audit
artifact. Source videos and corpus metadata are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2

if __package__:
    from scripts.render_dense_blind_followup import make_dense_sheet
    from scripts.score_visual_scene_baselines import resolve_clip, sample_frames
else:
    from render_dense_blind_followup import make_dense_sheet
    from score_visual_scene_baselines import resolve_clip, sample_frames


FOLLOWUP = {"crop_review", "dense_review", "audio_or_dense_review"}


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


def unique(rows: list[dict[str, Any]], key: str, source: str) -> dict[Any, dict[str, Any]]:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{source} contains duplicate {key}")
    return result


def select_followups(
    candidates: list[dict[str, Any]],
    blind: list[dict[str, Any]],
    post: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidate_by_item = unique(candidates, "item_id", "candidate manifest")
    blind_by_index = unique(blind, "audit_index", "blind manifest")
    post_by_index = unique(post, "audit_index", "post-reveal review")
    if set(blind_by_index) != set(post_by_index):
        raise ValueError("blind and post-reveal coverage mismatch")
    selected = []
    for audit_index in sorted(blind_by_index):
        if post_by_index[audit_index].get("disposition") not in FOLLOWUP:
            continue
        blind_row = blind_by_index[audit_index]
        item_id = blind_row["item_id"]
        candidate = candidate_by_item.get(item_id)
        if candidate is None:
            raise ValueError(f"selected item absent from candidate manifest: {item_id}")
        if candidate.get("uid") != blind_row.get("uid"):
            raise ValueError(f"candidate identity mismatch at audit_index {audit_index}")
        selected.append((candidate, blind_row))
    if not selected:
        raise ValueError("post-reveal ledger selected no follow-up items")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=96)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite dense audit directory: {args.out}")
    if args.frames < 12 or args.frames % 12:
        raise SystemExit("--frames must be a positive multiple of 12")

    selected = select_followups(
        load_jsonl(args.candidates),
        load_jsonl(args.blind_manifest),
        load_jsonl(args.post),
    )
    args.out.mkdir(parents=True)
    sheets = args.out / "blind_dense_sheets"
    sheets.mkdir()
    output = []
    for candidate, blind_row in selected:
        source = resolve_clip(candidate, args.candidates)
        frames, media = sample_frames(source, args.frames)
        if len(frames) < 12:
            raise RuntimeError(
                f"{blind_row['item_id']} produced only {len(frames)} dense frames"
            )
        audit_index = int(blind_row["audit_index"])
        target = sheets / f"{audit_index:02d}.jpg"
        sheet = make_dense_sheet(frames, media["sampled_timestamps"], audit_index)
        if not cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {target}")
        output.append(
            {
                "audit_index": audit_index,
                "item_id": blind_row["item_id"],
                "uid": blind_row["uid"],
                "pillar": blind_row["pillar"],
                "sheet_path": str(target),
                "sheet_sha256": sha256(target),
                "media": media,
            }
        )

    manifest = args.out / "blind_dense_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output)
    )
    summary = {
        "kind": "commentary_title_event_dense_followup",
        "items": len(output),
        "frames_requested_per_item": args.frames,
        "source_blind_manifest_sha256": sha256(args.blind_manifest),
        "source_post_reveal_sha256": sha256(args.post),
        "dense_manifest_sha256": sha256(manifest),
        "semantic_metadata_emitted": False,
        "corpus_mutated": False,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
