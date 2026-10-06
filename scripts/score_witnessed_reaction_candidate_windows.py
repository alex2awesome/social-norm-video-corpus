#!/usr/bin/env python3
"""Expand clip-wide reaction-candidate proposals over the witnessed corpus.

This is an append-only shadow artifact generator.  It never edits metadata,
clips, database rows, or acceptance labels.  A proposed window means only that
the transcript contains intervention-like language; reactor identity and event
validity remain unresolved and require role-aware video review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from scripts.witnessed_reaction_candidate_scan import (
        candidate_scan_features,
        scan_reaction_candidates,
        serialize_candidates,
    )
else:
    from witnessed_reaction_candidate_scan import (
        candidate_scan_features,
        scan_reaction_candidates,
        serialize_candidates,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_segments(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    segments = value.get("segments") or []
    return segments if isinstance(segments, list) else []


def clip_segments(
    segments: list[dict[str, Any]], window_start: float, window_end: float
) -> list[dict[str, Any]]:
    rows = []
    for segment in segments:
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        if end < window_start or start > window_end:
            continue
        row = {
            "start": start,
            "end": end,
            "clip_start": start - window_start,
            "clip_end": end - window_start,
            "text": str(segment.get("text") or "").strip(),
        }
        if segment.get("speaker") is not None:
            row["speaker"] = segment["speaker"]
        rows.append(row)
    return rows


def score_metadata(
    metadata_path: Path,
    transcript_path: Path,
) -> list[dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    uid = str(metadata.get("video_id") or metadata_path.parent.name)
    transcript = load_segments(transcript_path)
    by_clip: dict[int, list[dict[str, Any]]] = {}
    for reaction in metadata.get("reactions") or []:
        try:
            clip_idx = int(reaction["clip_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        by_clip.setdefault(clip_idx, []).append(reaction)
    rows = []
    for clip_idx, reactions in sorted(by_clip.items()):
        windows = [
            reaction.get("clip_window")
            for reaction in reactions
            if isinstance(reaction.get("clip_window"), list)
            and len(reaction["clip_window"]) == 2
        ]
        if not windows:
            continue
        window_start = min(float(window[0]) for window in windows)
        window_end = max(float(window[1]) for window in windows)
        segments = clip_segments(transcript, window_start, window_end)
        selected_quote = " ".join(
            str(reaction.get("matched_text") or reaction.get("phrase") or "")
            for reaction in reactions
        ).strip()
        selected_boundaries = [
            float(reaction["start"]) - window_start
            for reaction in reactions
            if reaction.get("start") is not None
        ]
        selected_boundary = (
            min(selected_boundaries) if selected_boundaries else None
        )
        candidates = scan_reaction_candidates(
            segments,
            selected_quote=selected_quote,
        )
        features = candidate_scan_features(
            candidates,
            selected_boundary=selected_boundary,
        )
        rows.append(
            {
                "item_id": f"witnessed:{uid}:clip_{clip_idx}",
                "uid": uid,
                "clip_idx": clip_idx,
                "clip_name": f"clip_{clip_idx}.mp4",
                "metadata_path": str(metadata_path),
                "transcript_path": str(transcript_path),
                "clip_window_source_sec": [window_start, window_end],
                "selected_detector_quote": selected_quote,
                "selected_detector_boundary_clip_sec": selected_boundary,
                "features": features,
                "candidates": serialize_candidates(candidates),
                "proposal_disposition": (
                    "send_candidate_windows_to_role_aware_video_stage"
                    if features["candidate_scan.active_nonnegative"] > 0
                    else "abstain_no_lexical_candidate"
                ),
                "acceptance_label": None,
                "corpus_disposition": None,
                "policy": "proposal_only_shadow_non_destructive",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen output: {args.out}")

    metadata_paths = sorted(args.hits.glob("*/metadata.json"))
    if args.limit is not None:
        metadata_paths = metadata_paths[: args.limit]
    rows: list[dict[str, Any]] = []
    bad_metadata: list[str] = []
    for metadata_path in metadata_paths:
        uid = metadata_path.parent.name
        try:
            rows.extend(
                score_metadata(
                    metadata_path,
                    args.transcripts / f"{uid}.json",
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            bad_metadata.append(f"{metadata_path}: {type(exc).__name__}: {exc}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    mechanism_counts = Counter(
        mechanism
        for row in rows
        for candidate in row["candidates"]
        for mechanism in candidate["mechanisms"]
    )
    summary = {
        "schema_version": 1,
        "kind": "witnessed_clipwide_reaction_candidate_windows_v1",
        "metadata_files": len(metadata_paths),
        "clip_records": len(rows),
        "clips_with_candidates": sum(
            row["features"]["candidate_scan.active_nonnegative"] > 0
            for row in rows
        ),
        "candidate_windows": sum(len(row["candidates"]) for row in rows),
        "mechanism_counts": dict(sorted(mechanism_counts.items())),
        "bad_metadata_count": len(bad_metadata),
        "bad_metadata_examples": bad_metadata[:20],
        "output_sha256": sha256_file(args.out),
        "audit_evidence": {
            "source_disjoint_items": 24,
            "strict_bystander_recall": 1.0,
            "strict_bystander_precision": 1 / 3,
            "interpretation": "proposal generator only; insufficient as an acceptance LF",
        },
        "policy": "shadow_only_non_destructive",
        "corpus_mutated": False,
    }
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
