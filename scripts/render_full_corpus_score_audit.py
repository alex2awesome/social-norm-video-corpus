#!/usr/bin/env python3
"""Freeze a blind, source-disjoint audit of full-corpus activity-score bands.

The selected score band and feature values are written only to the sealed
selection artifact. Blind sheets contain temporal pixels and timestamps but no
norm, query, title, or score. This is a review-queue calibration tool, never a
keep/drop mechanism.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__:
    from scripts.score_visual_scene_baselines import resolve_clip, sample_frames
else:
    from score_visual_scene_baselines import resolve_clip, sample_frames


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_successes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        row["item_id"]: row
        for row in rows
        if row.get("error") is None and row.get("low_level")
    }


def percentile(sorted_values: list[float], value: float) -> float:
    if len(sorted_values) <= 1:
        return 0.5
    return bisect.bisect_right(sorted_values, value) / len(sorted_values)


def scored_candidates(
    manifest_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    excluded_uids: set[str] | None = None,
) -> list[dict[str, Any]]:
    scores = latest_successes(score_rows)
    excluded_uids = excluded_uids or set()
    candidates = [
        {**row, "_score": scores[row["item_id"]]}
        for row in manifest_rows
        if row["item_id"] in scores
        and row.get("source_exists", True)
        and row["uid"] not in excluded_uids
    ]
    if not candidates:
        raise ValueError("no successful score records join the manifest")
    motion_values = sorted(
        float(row["_score"]["low_level"]["motion_mean"])
        for row in candidates
    )
    histogram_values = sorted(
        float(row["_score"]["low_level"]["histogram_delta_mean"])
        for row in candidates
    )
    enriched = []
    for row in candidates:
        low_level = row["_score"]["low_level"]
        motion = float(low_level["motion_mean"])
        histogram = float(low_level["histogram_delta_mean"])
        activity = (
            percentile(motion_values, motion)
            + percentile(histogram_values, histogram)
        ) / 2
        record = {
            **row,
            "_motion_percentile": percentile(motion_values, motion),
            "_histogram_percentile": percentile(histogram_values, histogram),
            "_activity_percentile": activity,
        }
        enriched.append(record)
    return enriched


def select_score_bands(
    manifest_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    per_band: int,
    seed: str,
    excluded_uids: set[str] | None = None,
    stratify_fields: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    cells: dict[str, list[dict[str, Any]]] = {
        "low_activity": [],
        "middle_activity": [],
        "high_activity": [],
    }
    for record in scored_candidates(
        manifest_rows,
        score_rows,
        excluded_uids,
    ):
        activity = record["_activity_percentile"]
        if activity <= 0.20:
            cells["low_activity"].append(record)
        elif 0.40 <= activity <= 0.60:
            cells["middle_activity"].append(record)
        elif activity >= 0.80:
            cells["high_activity"].append(record)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used_uids: set[str] = set()
    for band in ("low_activity", "middle_activity", "high_activity"):
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in cells[band]:
            stratum = tuple(
                str(row.get(field) or "<missing>")
                for field in stratify_fields
            )
            grouped[stratum].append(row)
        strata = list(grouped)
        rng.shuffle(strata)
        for stratum in strata:
            rng.shuffle(grouped[stratum])
        chosen = []
        while strata and len(chosen) < per_band:
            next_strata = []
            for stratum in strata:
                rows = grouped[stratum]
                row = None
                while rows:
                    candidate = rows.pop()
                    if candidate["uid"] not in used_uids:
                        row = candidate
                        break
                if row is not None:
                    row["_stratum"] = {
                        field: row.get(field)
                        for field in stratify_fields
                    }
                    chosen.append(row)
                    used_uids.add(row["uid"])
                if rows:
                    next_strata.append(stratum)
                if len(chosen) == per_band:
                    break
            strata = next_strata
        if len(chosen) != per_band:
            raise ValueError(
                f"insufficient source-disjoint {band} candidates: "
                f"wanted {per_band}, found {len(chosen)}"
            )
        for row in chosen:
            row["_score_band"] = band
            selected.append(row)
    rng.shuffle(selected)
    return selected


def select_uniform_sample(
    manifest_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    total: int,
    seed: str,
    excluded_uids: set[str] | None = None,
) -> list[dict[str, Any]]:
    candidates = scored_candidates(
        manifest_rows,
        score_rows,
        excluded_uids,
    )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = []
    used_uids = set()
    for row in candidates:
        if row["uid"] in used_uids:
            continue
        activity = row["_activity_percentile"]
        if activity <= 0.20:
            row["_score_band"] = "lowest_quintile"
        elif activity <= 0.40:
            row["_score_band"] = "second_quintile"
        elif activity <= 0.60:
            row["_score_band"] = "middle_quintile"
        elif activity <= 0.80:
            row["_score_band"] = "fourth_quintile"
        else:
            row["_score_band"] = "highest_quintile"
        row["_stratum"] = {}
        selected.append(row)
        used_uids.add(row["uid"])
        if len(selected) == total:
            break
    if len(selected) != total:
        raise ValueError(
            f"insufficient source-disjoint candidates: wanted {total}, "
            f"found {len(selected)}"
        )
    return selected


def fit_frame(frame: np.ndarray, width: int = 320, height: int = 180) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (
            max(1, int(round(frame.shape[1] * scale))),
            max(1, int(round(frame.shape[0] * scale))),
        ),
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def make_item_sheet(
    frames: list[np.ndarray],
    timestamps: list[float | None],
    audit_index: int,
) -> np.ndarray:
    tiles = []
    for frame, timestamp in zip(frames, timestamps):
        tile = np.zeros((204, 320, 3), dtype=np.uint8)
        tile[24:] = fit_frame(frame)
        label = "?" if timestamp is None else f"{timestamp:.1f}s"
        cv2.putText(
            tile,
            label,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    while len(tiles) < 12:
        tiles.append(np.zeros((204, 320, 3), dtype=np.uint8))
    rows = [
        np.hstack(tiles[index : index + 4])
        for index in range(0, 12, 4)
    ]
    sheet = np.vstack(rows)
    cv2.putText(
        sheet,
        f"#{audit_index:02d}",
        (1140, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return sheet


def make_superpages(
    item_sheets: list[tuple[int, np.ndarray]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    pages = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for page_index in range(0, len(item_sheets), 4):
        batch = item_sheets[page_index : page_index + 4]
        panels = []
        for audit_index, sheet in batch:
            panel = cv2.resize(sheet, (960, 459))
            cv2.putText(
                panel,
                f"ITEM {audit_index:02d}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
        while len(panels) < 4:
            panels.append(np.zeros((459, 960, 3), dtype=np.uint8))
        page = np.vstack(panels)
        target = out_dir / f"page_{page_index // 4:02d}.jpg"
        cv2.imwrite(str(target), page, [cv2.IMWRITE_JPEG_QUALITY, 88])
        pages.append(
            {
                "path": str(target),
                "sha256": file_sha256(target),
                "audit_indices": [value[0] for value in batch],
            }
        )
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--selection-mode",
        choices=("activity_bands", "uniform"),
        default="activity_bands",
    )
    parser.add_argument("--per-band", type=int, default=6)
    parser.add_argument("--total", type=int)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--seed", default="full-corpus-score-audit-v1")
    parser.add_argument("--exclude-uids", type=Path)
    parser.add_argument("--stratify-field", action="append", default=[])
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite audit directory: {args.out}")
    if args.frames != 12:
        raise SystemExit("the frozen sheet layout requires --frames 12")

    manifest_rows = load_jsonl(args.manifest)
    excluded_uids = (
        {
            line.strip()
            for line in args.exclude_uids.read_text().splitlines()
            if line.strip()
        }
        if args.exclude_uids
        else set()
    )
    score_rows = [
        row
        for score_path in args.scores
        for row in load_jsonl(score_path)
    ]
    if args.selection_mode == "uniform":
        if not args.total or args.total < 1:
            raise SystemExit("--total must be positive for uniform selection")
        selected = select_uniform_sample(
            manifest_rows,
            score_rows,
            args.total,
            args.seed,
            excluded_uids=excluded_uids,
        )
    else:
        selected = select_score_bands(
            manifest_rows,
            score_rows,
            args.per_band,
            args.seed,
            excluded_uids=excluded_uids,
            stratify_fields=tuple(args.stratify_field),
        )
    args.out.mkdir(parents=True)
    sheets_dir = args.out / "blind_item_sheets"
    sheets_dir.mkdir()
    item_sheets = []
    blind_rows = []
    sealed_rows = []
    for audit_index, row in enumerate(selected):
        path = resolve_clip(row, args.manifest)
        frames, media = sample_frames(
            path,
            args.frames,
            row.get("media_start_sec"),
            row.get("media_end_sec"),
        )
        if len(frames) < 8:
            raise RuntimeError(
                f"{row['item_id']} produced only {len(frames)} of "
                f"{args.frames} frames"
            )
        sheet = make_item_sheet(
            frames,
            media["sampled_timestamps"],
            audit_index,
        )
        target = sheets_dir / f"{audit_index:02d}.jpg"
        cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        item_sheets.append((audit_index, sheet))
        blind_rows.append(
            {
                "audit_index": audit_index,
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": row["pillar"],
                "sheet_path": str(target),
                "sheet_sha256": file_sha256(target),
                "media": media,
            }
        )
        sealed_rows.append(
            {
                "audit_index": audit_index,
                "item_id": row["item_id"],
                "uid": row["uid"],
                "score_band": row["_score_band"],
                "activity_percentile": row["_activity_percentile"],
                "motion_percentile": row["_motion_percentile"],
                "histogram_percentile": row["_histogram_percentile"],
                "low_level": row["_score"]["low_level"],
                "norm": row.get("norm"),
                "query_source": row.get("query_source"),
                "found_by_query": row.get("found_by_query"),
                "stratum": row.get("_stratum", {}),
            }
        )
    pages = make_superpages(item_sheets, args.out / "blind_review_pages")
    (args.out / "blind_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind_rows)
    )
    (args.out / "sealed_selection.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed_rows)
    )
    (args.out / "pages.json").write_text(
        json.dumps(
            {
                "kind": "blind_full_corpus_score_band_audit",
                "selection_mode": args.selection_mode,
                "seed": args.seed,
                "source_disjoint": True,
                "excluded_prior_review_uids": len(excluded_uids),
                "stratify_fields": args.stratify_field,
                "score_hidden_during_review": True,
                "items": len(blind_rows),
                "pages": pages,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "items": len(blind_rows),
                "pages": len(pages),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
