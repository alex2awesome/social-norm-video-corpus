#!/usr/bin/env python3
"""Render blind temporal sheets for a frozen visual-ranker confirmation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from scripts.render_full_corpus_score_audit import (
        file_sha256,
        make_item_sheet,
        make_superpages,
    )
    from scripts.score_visual_scene_baselines import resolve_clip, sample_frames
else:
    from render_full_corpus_score_audit import (
        file_sha256,
        make_item_sheet,
        make_superpages,
    )
    from score_visual_scene_baselines import resolve_clip, sample_frames


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def make_dense_sheet(
    frames: list[np.ndarray],
    timestamps: list[float | None],
    audit_index: int,
) -> np.ndarray:
    panels = [
        make_item_sheet(
            frames[start : start + 12],
            timestamps[start : start + 12],
            audit_index,
        )
        for start in range(0, len(frames), 12)
    ]
    return np.vstack(panels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--audit-index", type=int, action="append")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if args.frames < 12 or args.frames % 12:
        raise SystemExit("--frames must be a positive multiple of 12")
    rows = read_jsonl(args.selection)
    for row_index, row in enumerate(rows):
        row.setdefault("audit_index", row_index)
    if args.audit_index:
        requested = set(args.audit_index)
        rows = [
            row for row in rows if int(row["audit_index"]) in requested
        ]
        found = {int(row["audit_index"]) for row in rows}
        if found != requested:
            raise SystemExit(
                f"missing requested audit indices: {sorted(requested - found)}"
            )
    args.out.mkdir(parents=True)
    item_dir = args.out / "blind_item_sheets"
    item_dir.mkdir()
    item_sheets = []
    rendered = []
    for row_index, row in enumerate(rows):
        audit_index = int(row.get("audit_index", row_index))
        clip = (
            Path(row["clip"])
            if row.get("clip")
            else resolve_clip(row, args.selection)
        )
        frames, media = sample_frames(clip, args.frames)
        if len(frames) < 8:
            raise RuntimeError(f"{row['item_id']}: only {len(frames)} frames")
        sheet = make_dense_sheet(
            frames, media["sampled_timestamps"], audit_index
        )
        target = item_dir / f"{audit_index:02d}.jpg"
        cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        item_sheets.append((audit_index, sheet))
        rendered.append(
            {
                **row,
                "audit_index": audit_index,
                "sheet": str(target),
                "sheet_sha256": file_sha256(target),
                "media": media,
            }
        )
    pages = (
        make_superpages(item_sheets, args.out / "blind_review_pages")
        if args.frames == 12
        else []
    )
    (args.out / "rendered_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rendered)
    )
    (args.out / "render_freeze.json").write_text(
        json.dumps(
            {
                "kind": "blind_visual_ranker_confirmation_render",
                "selection_sha256": file_sha256(args.selection),
                "items": len(rendered),
                "frames_per_item": args.frames,
                "pages": pages,
                "score_hidden": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"items": len(rendered), "pages": len(pages)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
