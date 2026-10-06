#!/usr/bin/env python3
"""Select and render a blind, source-disjoint strict-audit calibration wave.

Selection is balanced across pillar, pre-existing versus delta low-level
coverage, and activity band.  Semantic labels and score values are sealed away
from the blind contact sheets.  The script is append-only and never changes the
corpus or its routing state.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2

try:
    from scripts.render_full_corpus_score_audit import (
        file_sha256,
        make_item_sheet,
        make_superpages,
    )
    from scripts.score_visual_scene_baselines import sample_frames
except ModuleNotFoundError:
    from render_full_corpus_score_audit import (  # type: ignore[no-redef]
        file_sha256,
        make_item_sheet,
        make_superpages,
    )
    from score_visual_scene_baselines import sample_frames  # type: ignore[no-redef]


PILLARS = ("instructional", "witnessed", "commentary")
ORIGINS = ("baseline", "delta")
BANDS = ("low", "middle", "high")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A score shard may be observed while its final line is being
                # appended. The frozen selection simply excludes that record.
                continue
            if isinstance(row, dict):
                yield row


def successful_scores(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in iter_jsonl(path):
            if row.get("error") is None and row.get("item_id") and row.get("low_level"):
                result[str(row["item_id"])] = row
    return result


def percentile(values: list[float], value: float) -> float:
    return bisect.bisect_right(values, value) / len(values) if values else 0.5


def activity_band(value: float) -> str | None:
    if value <= 0.25:
        return "low"
    if 0.375 <= value <= 0.625:
        return "middle"
    if value >= 0.75:
        return "high"
    return None


def candidates(
    labels: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    joined = []
    for label in labels:
        item_id = str(label.get("item_id") or "")
        score = scores.get(item_id)
        if (
            label.get("pillar") not in PILLARS
            or not label.get("media_present")
            or label.get("manual_strict_complete")
            or score is None
        ):
            continue
        low = score["low_level"]
        try:
            motion = float(low["motion_mean"])
            histogram = float(low["histogram_delta_mean"])
        except (KeyError, TypeError, ValueError):
            continue
        joined.append(
            {
                **label,
                "_score": score,
                "_motion": motion,
                "_histogram": histogram,
                "_origin": "baseline" if label.get("low_level_visual_complete") else "delta",
            }
        )
    for pillar in PILLARS:
        for origin in ORIGINS:
            cell = [row for row in joined if row["pillar"] == pillar and row["_origin"] == origin]
            motions = sorted(row["_motion"] for row in cell)
            histograms = sorted(row["_histogram"] for row in cell)
            for row in cell:
                row["_motion_percentile"] = percentile(motions, row["_motion"])
                row["_histogram_percentile"] = percentile(histograms, row["_histogram"])
                row["_activity_percentile"] = (
                    row["_motion_percentile"] + row["_histogram_percentile"]
                ) / 2
                row["_activity_band"] = activity_band(row["_activity_percentile"])
    return [row for row in joined if row.get("_activity_band")]


def select(
    rows: list[dict[str, Any]],
    per_cell: int,
    seed: str,
    excluded_uids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    excluded_uids = excluded_uids or set()
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["pillar"], row["_origin"], row["_activity_band"])].append(row)
    selected = []
    used_uids: set[str] = set(excluded_uids)
    for pillar in PILLARS:
        for origin in ORIGINS:
            for band in BANDS:
                key = (pillar, origin, band)
                values = grouped[key]
                rng.shuffle(values)
                chosen = []
                for row in values:
                    if row["uid"] in used_uids:
                        continue
                    chosen.append(row)
                    used_uids.add(row["uid"])
                    if len(chosen) == per_cell:
                        break
                if len(chosen) != per_cell:
                    raise ValueError(f"{key}: wanted {per_cell}, found {len(chosen)}")
                selected.extend(chosen)
    rng.shuffle(selected)
    return selected


def media_bounds(row: dict[str, Any]) -> tuple[float | None, float | None]:
    if row["pillar"] != "commentary":
        return None, None
    try:
        return max(0.0, float(row["start_sec"]) - 12.0), float(row["end_sec"]) + 12.0
    except (TypeError, ValueError):
        return None, None


def render(
    root: Path,
    labels_path: Path,
    score_paths: list[Path],
    out: Path,
    per_cell: int,
    seed: str,
    exclude_selections: list[Path] | None = None,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite frozen calibration: {out}")
    rows = candidates(list(iter_jsonl(labels_path)), successful_scores(score_paths))
    excluded_uids = {
        str(row["uid"])
        for path in (exclude_selections or [])
        for row in iter_jsonl(path)
        if row.get("uid")
    }
    selected = select(rows, per_cell, seed, excluded_uids)
    out.mkdir(parents=True)
    sheets = out / "blind_item_sheets"
    sheets.mkdir()
    blind_rows = []
    sealed_rows = []
    item_sheets = []
    for audit_index, row in enumerate(selected):
        source = root / str(row["media_path"])
        start, end = media_bounds(row)
        frames, media = sample_frames(source, 12, start, end)
        if len(frames) < 8:
            raise RuntimeError(f"{row['item_id']}: only {len(frames)} decodable frames")
        sheet = make_item_sheet(frames, media["sampled_timestamps"], audit_index)
        target = sheets / f"{audit_index:03d}.jpg"
        cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 91])
        item_sheets.append((audit_index, sheet))
        blind_rows.append(
            {
                "audit_index": audit_index,
                "opaque_id": hashlib.sha256(
                    f"{seed}:{row['item_id']}".encode()
                ).hexdigest()[:16],
                "sheet_path": str(target.relative_to(out)),
                "sheet_sha256": file_sha256(target),
                "sampled_frames": len(frames),
                "sampled_timestamps": media.get("sampled_timestamps"),
            }
        )
        sealed_rows.append(
            {
                "audit_index": audit_index,
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": row["pillar"],
                "coverage_origin": row["_origin"],
                "activity_band": row["_activity_band"],
                "activity_percentile": row["_activity_percentile"],
                "low_level": row["_score"]["low_level"],
                "media_path": row["media_path"],
                "norm": row.get("norm"),
                "polarity": row.get("polarity"),
                "reaction_tag": row.get("reaction_tag"),
                "reaction_text": row.get("reaction_text"),
                "reactor_role": row.get("reactor_role"),
                "signal": row.get("signal"),
                "quote": row.get("quote"),
                "query_source": row.get("query_source"),
                "category": row.get("category"),
                "start_sec": row.get("start_sec"),
                "end_sec": row.get("end_sec"),
            }
        )
    pages = make_superpages(item_sheets, out / "blind_review_pages")
    (out / "blind_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind_rows)
    )
    (out / "sealed_selection.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sealed_rows)
    )
    summary = {
        "schema_version": 1,
        "kind": "strict_audit_calibration_v1",
        "items": len(selected),
        "per_cell": per_cell,
        "cells": len(PILLARS) * len(ORIGINS) * len(BANDS),
        "source_disjoint": len({row["uid"] for row in selected}) == len(selected),
        "semantic_metadata_blinded": True,
        "seed": seed,
        "excluded_uids": len(excluded_uids),
        "pages": pages,
        "labels_sha256": file_sha256(labels_path),
        "score_inputs": [
            {"path": str(path), "sha256": file_sha256(path)} for path in score_paths
        ],
        "automatic_acceptance": False,
        "automatic_rejection": False,
        "corpus_mutated": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=2)
    parser.add_argument("--seed", default="strict-audit-calibration-v1-20260811")
    parser.add_argument(
        "--exclude-selection",
        type=Path,
        action="append",
        default=[],
        help="prior sealed selection whose source UIDs must not be reused",
    )
    args = parser.parse_args()
    if args.per_cell < 1:
        raise SystemExit("--per-cell must be positive")
    summary = render(
        args.root.resolve(),
        args.labels,
        args.scores,
        args.out,
        args.per_cell,
        args.seed,
        args.exclude_selection,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
