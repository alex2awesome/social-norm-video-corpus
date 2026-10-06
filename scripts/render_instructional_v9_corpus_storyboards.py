#!/usr/bin/env python3
"""Render resumable, metadata-blind dense storyboards for instructional V9.

The source manifest is used only to resolve clips and stable item identities.
No title, category, norm, polarity, transcript, or model output is rendered or
copied into the output manifest. Existing JPEGs are reusable after interruption;
the final manifest is written atomically only after exact source coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2

if __package__:
    from scripts.render_dense_blind_followup import make_dense_sheet
    from scripts.score_visual_scene_baselines import resolve_clip, sample_frames
else:
    from render_dense_blind_followup import make_dense_sheet
    from score_visual_scene_baselines import resolve_clip, sample_frames


FORBIDDEN_OUTPUT_FIELDS = {
    "title",
    "category",
    "norm",
    "polarity",
    "start_quote",
    "end_quote",
    "explanation",
    "found_by_query",
    "query_source",
}


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


def opaque_name(index: int) -> str:
    return f"{index:06d}.jpg"


def public_record(
    row: dict[str, Any],
    index: int,
    sheet: Path,
    sheet_sha256: str,
    media: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "storyboard_index": index,
        "item_id": str(row["item_id"]),
        "uid": str(row["uid"]),
        "pillar": str(row.get("pillar") or "instructional"),
        "sheet_path": str(sheet),
        "sheet_sha256": sheet_sha256,
        "frame_count": int(media["sampled_frames"]),
        "sampling_fps": media["fps"],
        "sampled_timestamps": media["sampled_timestamps"],
        "frame_order": "left_to_right_then_top_to_bottom",
    }
    leaked = FORBIDDEN_OUTPUT_FIELDS & record.keys()
    if leaked:
        raise AssertionError(f"semantic fields leaked into storyboard: {leaked}")
    return record


def failure_record(row: dict[str, Any], index: int, exc: Exception) -> dict[str, Any]:
    """Represent an undecodable source without leaking semantic metadata."""
    return {
        "storyboard_index": index,
        "item_id": str(row["item_id"]),
        "uid": str(row["uid"]),
        "pillar": str(row.get("pillar") or "instructional"),
        "sheet_path": None,
        "sheet_sha256": None,
        "frame_count": 0,
        "sampling_fps": None,
        "sampled_timestamps": [],
        "frame_order": "left_to_right_then_top_to_bottom",
        "error": f"{type(exc).__name__}: {exc}",
    }


def render_one(
    row: dict[str, Any],
    index: int,
    manifest: Path,
    sheets: Path,
    frame_count: int,
) -> dict[str, Any]:
    clip = resolve_clip(row, manifest)
    if not clip.is_file():
        raise FileNotFoundError(f"{row['item_id']}: clip missing: {clip}")
    frames, media = sample_frames(clip, frame_count)
    # Preserve every short clip rather than silently excluding it. Missing
    # cells remain blank and frame_count exposes the sparse temporal evidence
    # to downstream fail-closed scoring; neighboring frames are never copied.
    if not frames:
        raise RuntimeError(
            f"{row['item_id']}: no frames decoded"
        )
    target = sheets / opaque_name(index)
    if not target.is_file():
        rendered = make_dense_sheet(
            frames,
            media["sampled_timestamps"],
            index,
        )
        temporary = target.with_suffix(".tmp.jpg")
        if not cv2.imwrite(
            str(temporary),
            rendered,
            [cv2.IMWRITE_JPEG_QUALITY, 91],
        ):
            raise RuntimeError(f"failed to write {temporary}")
        os.replace(temporary, target)
    return public_record(row, index, target, sha256(target), media)


def validate_source(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("source manifest is empty")
    item_ids = [str(row.get("item_id") or "") for row in rows]
    if any(not item_id for item_id in item_ids):
        raise ValueError("source manifest contains a missing item_id")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("source manifest contains duplicate item_id values")
    if any(not row.get("uid") for row in rows):
        raise ValueError("source manifest contains a missing uid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.out_manifest.exists():
        raise SystemExit(f"refusing to overwrite: {args.out_manifest}")
    if args.frames < 12 or args.frames % 12:
        raise SystemExit("--frames must be a positive multiple of 12")

    rows = load_jsonl(args.manifest)
    validate_source(rows)
    sheets = args.out_dir / "storyboards"
    sheets.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                render_one,
                row,
                index,
                args.manifest,
                sheets,
                args.frames,
            ): (index, row)
            for index, row in enumerate(rows)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index, row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(failure_record(row, index, exc))
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"{completed}/{len(futures)} "
                    f"{row['item_id']}",
                    flush=True,
                )

    results.sort(key=lambda row: row["storyboard_index"])
    if {row["item_id"] for row in results} != {
        str(row["item_id"]) for row in rows
    }:
        raise RuntimeError("rendered storyboard identities do not exactly cover source")
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.out_manifest.with_suffix(".tmp.jsonl")
    with temporary_manifest.open("w") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary_manifest, args.out_manifest)
    summary = {
        "kind": "instructional_v9_corpus_storyboards",
        "items": len(results),
        "successfully_rendered_items": sum(not row.get("error") for row in results),
        "failed_render_items": sum(bool(row.get("error")) for row in results),
        "frames_requested_per_item": args.frames,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "storyboard_manifest": str(args.out_manifest.resolve()),
        "storyboard_manifest_sha256": sha256(args.out_manifest),
        "semantic_metadata_emitted": False,
        "corpus_mutated": False,
    }
    summary_path = args.out_manifest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
