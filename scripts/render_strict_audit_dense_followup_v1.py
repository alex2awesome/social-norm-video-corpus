#!/usr/bin/env python3
"""Render dense, metadata-blind follow-up for a strict calibration wave."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2

try:
    from scripts.render_dense_blind_followup import file_sha256, make_dense_sheet
    from scripts.score_visual_scene_baselines import sample_frames
except ModuleNotFoundError:
    from render_dense_blind_followup import file_sha256, make_dense_sheet  # type: ignore[no-redef]
    from score_visual_scene_baselines import sample_frames  # type: ignore[no-redef]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bounds(row: dict[str, Any]) -> tuple[float | None, float | None]:
    if row["pillar"] != "commentary":
        return None, None
    try:
        return max(0.0, float(row["start_sec"]) - 12), float(row["end_sec"]) + 12
    except (TypeError, ValueError):
        return None, None


def render(
    root: Path,
    selection: Path,
    reviews: Path,
    out: Path,
    controls: set[int],
    frames_requested: int,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite dense follow-up: {out}")
    selected = {int(row["audit_index"]): row for row in load_jsonl(selection)}
    reviewed = {int(row["audit_index"]): row for row in load_jsonl(reviews)}
    if set(selected) != set(reviewed):
        raise ValueError("manual blind review must cover the selection exactly")
    indices = sorted(
        index for index, row in reviewed.items()
        if row["dense_review_required"] == "yes" or index in controls
    )
    out.mkdir(parents=True)
    sheets = out / "blind_dense_sheets"
    sheets.mkdir()
    output = []
    for index in indices:
        row = selected[index]
        start, end = bounds(row)
        frames, media = sample_frames(
            root / row["media_path"], frames_requested, start, end
        )
        if len(frames) < 12:
            raise RuntimeError(f"audit index {index}: only {len(frames)} frames")
        sheet = make_dense_sheet(frames, media["sampled_timestamps"], index)
        target = sheets / f"{index:03d}.jpg"
        if not cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 91]):
            raise RuntimeError(f"failed to write {target}")
        output.append(
            {
                "audit_index": index,
                "opaque_id": hashlib.sha256(
                    f"strict-dense-v1:{row['item_id']}".encode()
                ).hexdigest()[:16],
                "control": index in controls,
                "sheet_path": str(target.relative_to(out)),
                "sheet_sha256": file_sha256(target),
                "sampled_frames": len(frames),
                "sampled_timestamps": media.get("sampled_timestamps"),
            }
        )
    manifest = out / "blind_dense_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output))
    summary = {
        "schema_version": 1,
        "kind": "strict_audit_dense_followup_v1",
        "items": len(output),
        "control_indices": sorted(controls),
        "frames_requested": frames_requested,
        "semantic_metadata_emitted": False,
        "source_selection_sha256": file_sha256(selection),
        "source_review_sha256": file_sha256(reviews),
        "manifest_sha256": file_sha256(manifest),
        "corpus_mutated": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control-index", type=int, action="append", default=[])
    parser.add_argument("--frames", type=int, default=36)
    args = parser.parse_args()
    if args.frames < 12 or args.frames % 12:
        raise SystemExit("--frames must be a positive multiple of 12")
    summary = render(
        args.root.resolve(), args.selection, args.reviews, args.out,
        set(args.control_index), args.frames,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
