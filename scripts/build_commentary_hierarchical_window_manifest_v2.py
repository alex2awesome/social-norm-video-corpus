#!/usr/bin/env python3
"""Build a diverse, auditable full-source commentary localization manifest.

Earlier commentary experiments trusted one VLM-proposed temporal interval and
then tried to repair that interval spatially.  This builder instead tiles each
retained source completely, annotates windows with transcript/VLM anchors, and
uses several cheap ranking signals only to choose a bounded set for exact VLM
and human review.  No ranking signal is an acceptance label.

Input source packets are JSONL rows with:

``uid, item_id, source_path, duration_sec, title, action_label``

and optional ``anchors`` entries of the form
``{"start_sec": ..., "end_sec": ..., "kind": ...}``.  Anchor kinds are
``prior_vlm``, ``commentary_statement``, and ``visual_deixis``.

The optional score JSONL must exactly cover every generated window and provide
numeric ``motion``, ``scene_change``, ``person_interaction``, and
``action_text_similarity`` values (or an explicit error, which fails closed to
the bottom of feature rankings).  Selection is deterministic and diversified
across anchor and feature rankings.  Every selected window remains an
unapproved candidate requiring VLM-output and manual visual audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ANCHOR_KINDS = {"prior_vlm", "commentary_statement", "visual_deixis"}
SCORE_FIELDS = (
    "motion",
    "scene_change",
    "person_interaction",
    "social_scene_similarity",
)
RANKING_ROUTES = (
    "anchor_visual_deixis",
    "anchor_commentary_statement",
    "anchor_prior_vlm",
    "feature_social_scene_similarity",
    "feature_person_interaction",
    "feature_motion",
    "feature_scene_change",
    "uniform_temporal_coverage",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )


def stable_id(uid: str, start: float, end: float) -> str:
    payload = f"{uid}\0{start:.3f}\0{end:.3f}".encode()
    return "commentary-window-" + hashlib.sha256(payload).hexdigest()[:20]


def validate_sources(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("source packet is empty")
    uids: set[str] = set()
    item_ids: set[str] = set()
    for row in rows:
        uid = str(row.get("uid") or "")
        item_id = str(row.get("item_id") or "")
        if not uid or uid in uids:
            raise ValueError("source packets have missing or duplicate uid")
        if not item_id or item_id in item_ids:
            raise ValueError("source packets have missing or duplicate item_id")
        uids.add(uid)
        item_ids.add(item_id)
        duration = row.get("duration_sec")
        if not isinstance(duration, (int, float)) or not 0 < float(duration) <= 1800:
            raise ValueError(f"{uid}: invalid or over-limit duration_sec")
        for field in ("source_path", "title", "action_label"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{uid}: missing {field}")
        for anchor in row.get("anchors") or []:
            if anchor.get("kind") not in ANCHOR_KINDS:
                raise ValueError(f"{uid}: invalid anchor kind")
            start = anchor.get("start_sec")
            end = anchor.get("end_sec")
            if (
                not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not 0 <= float(start) < float(end) <= float(duration) + 1e-6
            ):
                raise ValueError(f"{uid}: invalid anchor bounds")


def tiled_bounds(
    duration: float, window_sec: float = 12.0, stride_sec: float = 8.0
) -> list[tuple[float, float]]:
    """Cover the complete source, including a tail-aligned final window."""
    if duration <= 0 or window_sec <= 0 or stride_sec <= 0 or stride_sec > window_sec:
        raise ValueError("invalid tiling parameters")
    if duration <= window_sec:
        return [(0.0, round(duration, 3))]
    starts: list[float] = []
    current = 0.0
    while current + window_sec < duration:
        starts.append(current)
        current += stride_sec
    tail = max(0.0, duration - window_sec)
    if not starts or abs(starts[-1] - tail) > 1e-6:
        starts.append(tail)
    return [(round(start, 3), round(min(duration, start + window_sec), 3))
            for start in starts]


def overlaps(start: float, end: float, anchor: dict[str, Any]) -> bool:
    return float(anchor["start_sec"]) < end and float(anchor["end_sec"]) > start


def build_windows(
    sources: list[dict[str, Any]],
    *,
    window_sec: float = 12.0,
    stride_sec: float = 8.0,
) -> list[dict[str, Any]]:
    validate_sources(sources)
    output: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda row: str(row["uid"])):
        uid = str(source["uid"])
        duration = float(source["duration_sec"])
        anchors = source.get("anchors") or []
        for ordinal, (start, end) in enumerate(
            tiled_bounds(duration, window_sec, stride_sec)
        ):
            kinds = sorted({
                str(anchor["kind"]) for anchor in anchors
                if overlaps(start, end, anchor)
            })
            window_id = stable_id(uid, start, end)
            output.append({
                "window_id": window_id,
                "uid": uid,
                # The existing visual baseline scorer resumes and shards by
                # item_id, so each bounded window must be its own item while
                # retaining explicit source lineage separately.
                "item_id": window_id,
                "source_item_id": source["item_id"],
                "source_path": source["source_path"],
                "pillar": "commentary",
                "source_duration_sec": duration,
                "title": source["title"],
                "action_label": source["action_label"],
                "window_ordinal": ordinal,
                "window_start_sec": start,
                "window_end_sec": end,
                "media_start_sec": start,
                "media_end_sec": end,
                "anchor_kinds": kinds,
                "candidate_generation_only": True,
                "automatic_acceptance": False,
                "corpus_mutation_authorized": False,
            })
    ids = [row["window_id"] for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("generated duplicate window_id")
    return output


def validate_full_coverage(windows: list[dict[str, Any]]) -> None:
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windows:
        by_uid[str(row["uid"])].append(row)
    for uid, rows in by_uid.items():
        ordered = sorted(rows, key=lambda row: float(row["window_start_sec"]))
        duration = float(ordered[0]["source_duration_sec"])
        if float(ordered[0]["window_start_sec"]) != 0:
            raise ValueError(f"{uid}: tiling does not begin at zero")
        covered_end = 0.0
        for row in ordered:
            start = float(row["window_start_sec"])
            end = float(row["window_end_sec"])
            if start > covered_end + 1e-6:
                raise ValueError(f"{uid}: tiling has a gap before {start}")
            covered_end = max(covered_end, end)
        if abs(covered_end - duration) > 1e-3:
            raise ValueError(f"{uid}: tiling does not cover source tail")


def index_scores(
    windows: list[dict[str, Any]], scores: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    expected = {str(row["window_id"]) for row in windows}
    indexed = {str(row.get("window_id") or ""): row for row in scores}
    if len(indexed) != len(scores) or set(indexed) != expected:
        raise ValueError("cheap feature scores must exactly cover generated windows")
    for window_id, row in indexed.items():
        if row.get("error"):
            continue
        for field in SCORE_FIELDS:
            value = row.get(field)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{window_id}: invalid or missing {field}")
    return indexed


def _numeric_score(row: dict[str, Any], field: str) -> float:
    return float("-inf") if row.get("error") else float(row[field])


def _uniform_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Farthest-point-like deterministic coverage ordering over ordinals."""
    ordered = sorted(rows, key=lambda row: int(row["window_ordinal"]))
    if len(ordered) <= 2:
        return ordered
    positions = [0, len(ordered) - 1, len(ordered) // 2]
    remaining = [index for index in range(len(ordered)) if index not in positions]
    while remaining:
        index = max(
            remaining,
            key=lambda value: (
                min(abs(value - chosen) for chosen in positions), -value
            ),
        )
        positions.append(index)
        remaining.remove(index)
    return [ordered[index] for index in positions]


def select_diverse(
    windows: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    *,
    per_route: int = 2,
    max_per_source: int = 14,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if per_route <= 0 or max_per_source <= 0:
        raise ValueError("selection limits must be positive")
    validate_full_coverage(windows)
    score_index = index_scores(windows, scores)
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windows:
        by_uid[str(row["uid"])].append(row)

    selected: list[dict[str, Any]] = []
    route_counts: dict[str, int] = defaultdict(int)
    score_errors = sum(bool(row.get("error")) for row in scores)
    for uid in sorted(by_uid):
        rows = by_uid[uid]
        rankings: dict[str, list[dict[str, Any]]] = {
            "anchor_visual_deixis": sorted(
                (row for row in rows if "visual_deixis" in row["anchor_kinds"]),
                key=lambda row: int(row["window_ordinal"]),
            ),
            "anchor_commentary_statement": sorted(
                (row for row in rows if "commentary_statement" in row["anchor_kinds"]),
                key=lambda row: int(row["window_ordinal"]),
            ),
            "anchor_prior_vlm": sorted(
                (row for row in rows if "prior_vlm" in row["anchor_kinds"]),
                key=lambda row: int(row["window_ordinal"]),
            ),
            "uniform_temporal_coverage": _uniform_order(rows),
        }
        for field in SCORE_FIELDS:
            rankings[f"feature_{field}"] = sorted(
                rows,
                key=lambda row: (
                    -_numeric_score(score_index[row["window_id"]], field),
                    int(row["window_ordinal"]),
                ),
            )

        picked: dict[str, dict[str, Any]] = {}
        reasons: dict[str, list[str]] = defaultdict(list)
        # Round-robin prevents a long anchor list or one feature from consuming
        # the entire source budget before other mechanisms are represented.
        for rank in range(per_route):
            for route in RANKING_ROUTES:
                if len(picked) >= max_per_source:
                    break
                candidates = rankings[route]
                if rank >= len(candidates):
                    continue
                row = candidates[rank]
                window_id = str(row["window_id"])
                reasons[window_id].append(route)
                picked.setdefault(window_id, row)
            if len(picked) >= max_per_source:
                break
        for window_id, row in sorted(
            picked.items(), key=lambda value: int(value[1]["window_ordinal"])
        ):
            output = dict(row)
            output["selection_reasons"] = sorted(reasons[window_id])
            output["cheap_features"] = {
                field: (
                    None if score_index[window_id].get("error")
                    else float(score_index[window_id][field])
                )
                for field in SCORE_FIELDS
            }
            output["cheap_feature_error"] = score_index[window_id].get("error")
            output["vlm_output_manual_audit_required"] = True
            output["exact_visual_clip_manual_audit_required"] = True
            selected.append(output)
            for route in output["selection_reasons"]:
                route_counts[route] += 1

    return selected, {
        "kind": "commentary_hierarchical_full_source_window_selection_v2",
        "sources": len(by_uid),
        "generated_windows": len(windows),
        "selected_windows": len(selected),
        "per_route": per_route,
        "max_per_source": max_per_source,
        "cheap_feature_score_errors": score_errors,
        "cheap_feature_score_coverage": (
            (len(scores) - score_errors) / len(windows) if windows else None
        ),
        "selection_reason_counts": dict(sorted(route_counts.items())),
        "full_source_tiling_verified": True,
        "all_selected_vlm_outputs_require_manual_audit": True,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--all-windows-out", type=Path, required=True)
    parser.add_argument("--selected-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=12.0)
    parser.add_argument("--stride-sec", type=float, default=8.0)
    parser.add_argument("--per-route", type=int, default=2)
    parser.add_argument("--max-per-source", type=int, default=14)
    args = parser.parse_args()
    windows = build_windows(
        read_jsonl(args.sources),
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
    )
    selected, summary = select_diverse(
        windows,
        read_jsonl(args.scores),
        per_route=args.per_route,
        max_per_source=args.max_per_source,
    )
    write_jsonl(args.all_windows_out, windows)
    write_jsonl(args.selected_out, selected)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
