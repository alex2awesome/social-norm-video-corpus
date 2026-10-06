#!/usr/bin/env python3
"""Render a metadata-blind dense audit of every unaudited title-cue source."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def blind_rank(seed: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{item_id}".encode()).hexdigest()


def select_remaining(
    candidates: list[dict[str, Any]],
    audited: list[dict[str, Any]],
    seed: str,
) -> list[dict[str, Any]]:
    if len({row["item_id"] for row in candidates}) != len(candidates):
        raise ValueError("candidate manifest contains duplicate item_id")
    audited_ids = {row["item_id"] for row in audited}
    if not audited_ids <= {row["item_id"] for row in candidates}:
        raise ValueError("audited selection is not a subset of candidates")
    selected = [
        dict(row) for row in candidates if row["item_id"] not in audited_ids
    ]
    selected.sort(key=lambda row: blind_rank(seed, row["item_id"]))
    for audit_index, row in enumerate(selected):
        row["audit_index"] = audit_index
        row["candidate_id"] = f"commentary-title-pop-{audit_index:04d}"
    return selected


def render_one(
    row: dict[str, Any],
    candidates_path: Path,
    sheets: Path,
    frames_requested: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = resolve_clip(row, candidates_path)
    frames, media = sample_frames(source, frames_requested)
    if len(frames) < 12:
        raise RuntimeError(
            f"{row['candidate_id']} produced only {len(frames)} frames"
        )
    target = sheets / f"{row['candidate_id']}.jpg"
    sheet = make_dense_sheet(frames, media["sampled_timestamps"], row["audit_index"])
    if target.exists():
        existing = cv2.imread(str(target))
        if existing is None or existing.shape != sheet.shape:
            raise RuntimeError(f"existing sheet is invalid: {target}")
    elif not cv2.imwrite(
        str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]
    ):
        raise RuntimeError(f"failed to write {target}")
    semantic = {
        **row,
        "source_path_resolved": str(source),
        "sheet_path": str(target),
        "sheet_sha256": sha256(target),
        "media": media,
    }
    blind = {
        "audit_index": row["audit_index"],
        "candidate_id": row["candidate_id"],
        "sheet_path": str(target),
        "sheet_sha256": semantic["sheet_sha256"],
        "sampled_timestamps": media["sampled_timestamps"],
        "frame_count": len(frames),
    }
    return semantic, blind


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--audited-selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--blind-seed",
        default="20260728-commentary-title-population-dense-v1",
    )
    args = parser.parse_args()
    if args.out.exists():
        existing = {path.name for path in args.out.iterdir()}
        allowed_resume = {"README.md", "blind_dense_sheets"}
        if existing - allowed_resume:
            raise SystemExit(f"refusing to overwrite: {args.out}")
    if args.frames < 12 or args.frames % 12:
        raise SystemExit("--frames must be a multiple of 12")

    candidates = load_jsonl(args.candidates)
    audited = load_jsonl(args.audited_selection)
    selected = select_remaining(candidates, audited, args.blind_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    sheets = args.out / "blind_dense_sheets"
    sheets.mkdir(exist_ok=True)
    rendered: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                render_one,
                row,
                args.candidates,
                sheets,
                args.frames,
            ): row
            for row in selected
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                rendered[row["audit_index"]] = future.result()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                rendered[row["audit_index"]] = (
                    {**row, "error": error},
                    {
                        "audit_index": row["audit_index"],
                        "candidate_id": row["candidate_id"],
                        "sheet_path": None,
                        "sheet_sha256": None,
                        "sampled_timestamps": [],
                        "frame_count": 0,
                        "error": error,
                    },
                )
            if index % 10 == 0:
                print(f"{index}/{len(futures)} rendered", flush=True)

    semantic_rows = [rendered[index][0] for index in sorted(rendered)]
    blind_rows = [rendered[index][1] for index in sorted(rendered)]
    semantic_path = args.out / "semantic_selection.jsonl"
    blind_path = args.out / "blind_manifest.jsonl"
    semantic_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in semantic_rows)
    )
    blind_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind_rows)
    )
    summary = {
        "kind": "commentary_title_population_dense_v1",
        "candidate_population_items": len(candidates),
        "previously_audited_items": len(audited),
        "exhaustive_remaining_items": len(selected),
        "successfully_rendered_items": sum(
            row.get("error") is None for row in blind_rows
        ),
        "failed_render_items": sum(
            row.get("error") is not None for row in blind_rows
        ),
        "frames_requested_per_item": args.frames,
        "blind_seed": args.blind_seed,
        "semantic_selection_sha256": sha256(semantic_path),
        "blind_manifest_sha256": sha256(blind_path),
        "semantic_metadata_in_blind_manifest": False,
        "corpus_mutated": False,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
