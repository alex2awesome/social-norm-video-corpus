#!/usr/bin/env python3
"""Render metadata-blind dense sheets for explicit time/crop audit plans.

Each JSONL plan row supplies a source path, a bounded time interval, and an
optional normalized crop box ``[x0, y0, x1, y1]``. The output is derived and
read-only with respect to source videos and corpus metadata.
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
    from scripts.render_dense_blind_followup import make_dense_sheet
    from scripts.score_visual_scene_baselines import sample_frames
else:
    from render_dense_blind_followup import make_dense_sheet
    from score_visual_scene_baselines import sample_frames


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


def validate_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    for row in rows:
        variant = str(row["variant_id"])
        if variant in seen:
            raise ValueError(f"duplicate variant_id: {variant}")
        seen.add(variant)
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        if not 0 <= start < end:
            raise ValueError(f"invalid time bounds for {variant}")
        crop = row.get("crop_norm", [0.0, 0.0, 1.0, 1.0])
        if (
            not isinstance(crop, list)
            or len(crop) != 4
            or not all(isinstance(value, (int, float)) for value in crop)
        ):
            raise ValueError(f"invalid crop for {variant}")
        x0, y0, x1, y1 = map(float, crop)
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"invalid crop bounds for {variant}")
        source = Path(row["source_path"])
        if not source.is_file():
            raise ValueError(f"missing source for {variant}: {source}")
    if not rows:
        raise ValueError("empty plan")
    return rows


def crop_frames(
    frames: list[np.ndarray],
    crop_norm: list[float],
) -> tuple[list[np.ndarray], list[int]]:
    cropped = []
    pixels: list[int] | None = None
    for frame in frames:
        height, width = frame.shape[:2]
        x0 = int(round(float(crop_norm[0]) * width))
        y0 = int(round(float(crop_norm[1]) * height))
        x1 = int(round(float(crop_norm[2]) * width))
        y1 = int(round(float(crop_norm[3]) * height))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 - x0 < 32 or y1 - y0 < 32:
            raise ValueError("crop is smaller than 32 pixels in one dimension")
        pixels = [x0, y0, x1, y1]
        cropped.append(frame[y0:y1, x0:x1].copy())
    if pixels is None:
        raise ValueError("cannot crop an empty frame list")
    return cropped, pixels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite audit directory: {args.out}")
    if args.frames < 12 or args.frames % 12:
        raise SystemExit("--frames must be a positive multiple of 12")
    rows = validate_plan(load_jsonl(args.plan))

    args.out.mkdir(parents=True)
    sheets = args.out / "blind_sheets"
    sheets.mkdir()
    output = []
    for row in rows:
        source = Path(row["source_path"])
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        frames, media = sample_frames(source, args.frames, start, end)
        if len(frames) < 8:
            raise RuntimeError(
                f"{row['variant_id']} produced only {len(frames)} frames"
            )
        crop_norm = list(map(float, row.get("crop_norm", [0, 0, 1, 1])))
        frames, crop_pixels = crop_frames(frames, crop_norm)
        audit_index = int(row["audit_index"])
        sheet = make_dense_sheet(
            frames,
            media["sampled_timestamps"],
            audit_index,
        )
        target = sheets / f"{row['variant_id']}.jpg"
        if not cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 93]):
            raise RuntimeError(f"failed to write {target}")
        output.append(
            {
                "audit_index": audit_index,
                "item_id": row["item_id"],
                "uid": row["uid"],
                "variant_id": row["variant_id"],
                "time_bounds_sec": [start, end],
                "crop_norm": crop_norm,
                "crop_pixels": crop_pixels,
                "sheet_path": str(target),
                "sheet_sha256": sha256(target),
                "media": media,
                "semantic_metadata_emitted": False,
            }
        )

    manifest = args.out / "blind_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output)
    )
    summary = {
        "kind": "bounded_crop_followup",
        "variants": len(output),
        "frames_requested_per_variant": args.frames,
        "plan_sha256": sha256(args.plan),
        "manifest_sha256": sha256(manifest),
        "semantic_metadata_emitted": False,
        "corpus_mutated": False,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
