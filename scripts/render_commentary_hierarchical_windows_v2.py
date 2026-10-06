#!/usr/bin/env python3
"""Render selected commentary windows for blind human and VLM review.

This is a thin lineage-preserving adapter around the established temporal
storyboard renderer. It indexes retained full sources with symlinks (no media
copy), renders each selected bounded interval, preserves unmasked pages for the
human first pass, and emits OCR-masked pages for the VLM. The selected source
manifest remains authoritative; this script never changes source media.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

if __package__:
    from scripts.build_commentary_hierarchical_window_manifest_v2 import (
        read_jsonl,
        write_jsonl,
    )
    from scripts.build_commentary_temporal_verifier_manifest import build_rows
else:
    from build_commentary_hierarchical_window_manifest_v2 import read_jsonl, write_jsonl
    from build_commentary_temporal_verifier_manifest import build_rows


BLIND_FIELDS = (
    "audit_index",
    "window_id",
    "uid",
    "performed_event_visible",
    "actor_target_grounded",
    "before_action_after_complete",
    "crucial_action_occluded_or_offframe",
    "label_bearing_text_absent",
    "literal_action_description",
    "manual_rationale",
)
POST_REVEAL_FIELDS = (
    "audit_index",
    "window_id",
    "uid",
    "exact_named_action_visible",
    "label_alignment",
    "start_boundary_clean",
    "end_boundary_clean",
    "commentary_visual_route",
    "vlm_output_visually_supported",
    "manual_rationale",
)


def prepare_specs(
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not selected:
        raise ValueError("selected commentary window manifest is empty")
    ids: set[str] = set()
    specs = []
    semantic = []
    lineage: dict[str, dict[str, Any]] = {}
    for audit_index, row in enumerate(sorted(
        selected,
        key=lambda value: (
            str(value.get("uid") or ""), int(value.get("window_ordinal") or 0)
        ),
    )):
        window_id = str(row.get("window_id") or "")
        uid = str(row.get("uid") or "")
        if not window_id or window_id in ids or not uid:
            raise ValueError("selected manifest has missing/duplicate window_id or uid")
        ids.add(window_id)
        if row.get("item_id") != window_id:
            raise ValueError(f"{window_id}: scorer item lineage mismatch")
        start = float(row["window_start_sec"])
        end = float(row["window_end_sec"])
        duration = float(row["source_duration_sec"])
        if not 0 <= start < end <= duration + 1e-6:
            raise ValueError(f"{window_id}: invalid selected bounds")
        source_path = str(row.get("source_path") or "")
        if not source_path:
            raise ValueError(f"{window_id}: missing source_path")
        specs.append({
            "audit_index": audit_index,
            "start_sec": start,
            "end_sec": end,
        })
        semantic.append({
            "audit_index": audit_index,
            "candidate_id": window_id,
            "uid": uid,
            "norm": row["action_label"],
        })
        lineage[window_id] = {
            "audit_index": audit_index,
            "window_id": window_id,
            "uid": uid,
            "source_item_id": row["source_item_id"],
            "source_path": source_path,
            "source_duration_sec": duration,
            "title": row["title"],
            "action_label": row["action_label"],
            "window_ordinal": int(row["window_ordinal"]),
            "window_start_sec": start,
            "window_end_sec": end,
            "selection_reasons": list(row.get("selection_reasons") or []),
            "cheap_features": dict(row.get("cheap_features") or {}),
            "cheap_feature_error": row.get("cheap_feature_error"),
        }
    return specs, semantic, lineage


def create_source_links(lineage: dict[str, dict[str, Any]], proxy_dir: Path) -> None:
    proxy_dir.mkdir(parents=True, exist_ok=True)
    for window_id, row in lineage.items():
        source = Path(row["source_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        link = proxy_dir / f"{window_id}.mp4"
        if link.is_symlink():
            if link.resolve() != source.resolve():
                raise ValueError(f"{window_id}: existing source link points elsewhere")
            continue
        if link.exists():
            raise FileExistsError(link)
        os.symlink(source.resolve(), link)


def manual_templates(lineage: dict[str, dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    blind = []
    post = []
    for window_id, row in sorted(lineage.items(), key=lambda value: value[1]["audit_index"]):
        common = {
            "audit_index": str(row["audit_index"]),
            "window_id": window_id,
            "uid": row["uid"],
        }
        blind.append({**common, **{
            field: "" for field in BLIND_FIELDS if field not in common
        }})
        post.append({**common, **{
            field: "" for field in POST_REVEAL_FIELDS if field not in common
        }})
    return blind, post


def write_tsv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--max-frames-per-window", type=int, default=36)
    parser.add_argument("--cell-width", type=int, default=256)
    parser.add_argument("--cell-max-height", type=int, default=256)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    selected = read_jsonl(args.selected)
    specs, semantic, lineage = prepare_specs(selected)
    proxy_dir = args.out_dir / "source_links"
    create_source_links(lineage, proxy_dir)
    manifest_path = args.out_dir / "sealed_render_manifest.jsonl"
    rendered = build_rows(
        specs,
        semantic,
        proxy_dir,
        args.out_dir / "rendered",
        args.out_dir,
        args.ffmpeg,
        fps=args.fps,
        columns=6,
        rows_per_page=6,
        cell_width=args.cell_width,
        cell_max_height=args.cell_max_height,
        ocr_mask=True,
        ocr_all_screen=True,
        max_frames_per_window=args.max_frames_per_window,
    )
    merged = []
    for row in rendered:
        window_id = str(row["candidate_id"])
        merged.append({**row, **lineage[window_id],
                       "policy": "manual_and_vlm_audit_only_no_acceptance"})
    write_jsonl(manifest_path, merged)
    blind, post = manual_templates(lineage)
    write_tsv(args.out_dir / "blind_manual_review.tsv", blind, BLIND_FIELDS)
    write_tsv(args.out_dir / "post_reveal_manual_review.tsv", post, POST_REVEAL_FIELDS)
    summary = {
        "kind": "commentary_hierarchical_window_render_v2",
        "selected_windows": len(selected),
        "rendered_windows": len(merged),
        "unmasked_human_pages_preserved": True,
        "ocr_masked_vlm_pages": True,
        "blind_manual_rows": len(blind),
        "post_reveal_manual_rows": len(post),
        "source_media_copied": False,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
