#!/usr/bin/env python3
"""Render dense, metadata-blind follow-up sheets from frozen manual reviews.

The corpus manifest is used only to resolve media paths and temporal bounds.
Semantic fields are never copied to the output. This script is observational:
it does not change labels, metadata, or source media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__:
    from scripts.render_full_corpus_score_audit import make_item_sheet
    from scripts.score_visual_scene_baselines import resolve_clip, sample_frames
    from scripts.validate_blind_visual_review import validate
else:
    from render_full_corpus_score_audit import make_item_sheet
    from score_visual_scene_baselines import resolve_clip, sample_frames
    from validate_blind_visual_review import validate


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_by_item_id(
    rows: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("item_id")
        if not item_id:
            raise ValueError(f"{source} contains a missing item_id")
        if item_id in result:
            raise ValueError(f"{source} contains duplicate item_id {item_id}")
        result[str(item_id)] = row
    return result


def select_dense_rows(
    corpus_rows: list[dict[str, Any]],
    blind_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    validate(blind_rows, review_rows)
    corpus = unique_by_item_id(corpus_rows, source="corpus manifest")
    blind = unique_by_item_id(blind_rows, source="blind manifest")
    reviews = unique_by_item_id(review_rows, source="manual reviews")
    selected = []
    for item_id, blind_row in sorted(
        blind.items(),
        key=lambda pair: int(pair[1]["audit_index"]),
    ):
        review = reviews[item_id]
        if review["dense_review_required"] != "yes":
            continue
        if item_id not in corpus:
            raise ValueError(f"blind item is absent from corpus manifest: {item_id}")
        selected.append((corpus[item_id], blind_row, review))
    if not selected:
        raise ValueError("manual reviews selected no dense follow-up items")
    return selected


def make_dense_sheet(
    frames: list[np.ndarray],
    timestamps: list[float | None],
    audit_index: int,
) -> np.ndarray:
    panels = []
    count = (len(frames) + 11) // 12
    for panel_index, start in enumerate(range(0, len(frames), 12)):
        panel = make_item_sheet(
            frames[start : start + 12],
            timestamps[start : start + 12],
            audit_index,
        )
        cv2.putText(
            panel,
            f"DENSE {panel_index + 1}/{count}",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    return np.vstack(panels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=36)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite dense audit directory: {args.out}")
    if args.frames < 12 or args.frames % 12:
        raise SystemExit("--frames must be a positive multiple of 12")

    corpus_rows = load_jsonl(args.manifest)
    blind_rows = load_jsonl(args.blind_manifest)
    review_rows = load_jsonl(args.reviews)
    selected = select_dense_rows(corpus_rows, blind_rows, review_rows)

    args.out.mkdir(parents=True)
    sheets_dir = args.out / "blind_dense_sheets"
    sheets_dir.mkdir()
    output_rows = []
    for corpus_row, blind_row, _review in selected:
        path = resolve_clip(corpus_row, args.manifest)
        frames, media = sample_frames(
            path,
            args.frames,
            corpus_row.get("media_start_sec"),
            corpus_row.get("media_end_sec"),
        )
        if len(frames) < 8:
            raise RuntimeError(
                f"{blind_row['item_id']} produced only {len(frames)} dense frames"
            )
        audit_index = int(blind_row["audit_index"])
        sheet = make_dense_sheet(
            frames,
            media["sampled_timestamps"],
            audit_index,
        )
        target = sheets_dir / f"{audit_index:02d}.jpg"
        if not cv2.imwrite(
            str(target),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 91],
        ):
            raise RuntimeError(f"failed to write dense sheet: {target}")
        output_rows.append(
            {
                "audit_index": audit_index,
                "item_id": blind_row["item_id"],
                "uid": blind_row["uid"],
                "pillar": blind_row["pillar"],
                "sheet_path": str(target),
                "sheet_sha256": file_sha256(target),
                "media": media,
            }
        )

    dense_manifest = args.out / "blind_dense_manifest.jsonl"
    dense_manifest.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in output_rows
        )
    )
    summary = {
        "kind": "dense_blind_followup",
        "items": len(output_rows),
        "frames_requested_per_item": args.frames,
        "source_blind_manifest": str(args.blind_manifest.resolve()),
        "source_blind_manifest_sha256": file_sha256(args.blind_manifest),
        "source_reviews": str(args.reviews.resolve()),
        "source_reviews_sha256": file_sha256(args.reviews),
        "dense_manifest": str(dense_manifest.resolve()),
        "dense_manifest_sha256": file_sha256(dense_manifest),
        "semantic_metadata_emitted": False,
        "corpus_mutated": False,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
