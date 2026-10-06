#!/usr/bin/env python3
"""Build unapproved exact-artifact plans from frozen manual localization gold.

These plans do not keep clips.  They identify what must be cut/muted/cropped
and then re-audited as a newly hashed artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def indexed(rows: list[dict[str, Any]], field: str, name: str) -> dict[str, dict[str, Any]]:
    output = {str(row.get(field) or ""): row for row in rows}
    if not output or not all(output) or len(output) != len(rows):
        raise ValueError(f"{name} has missing or duplicate {field}")
    return output


def storyboard_bounds(timestamps: list[float], start_index: int, end_index: int) -> tuple[float, float]:
    if not timestamps or not 0 <= start_index <= end_index < len(timestamps):
        raise ValueError("manual frame bounds exceed storyboard timestamps")
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:]) if right > left]
    typical_gap = statistics.median(gaps) if gaps else 0.25
    start = 0.0 if start_index == 0 else (
        timestamps[start_index - 1] + timestamps[start_index]
    ) / 2
    end = (
        (timestamps[end_index] + timestamps[end_index + 1]) / 2
        if end_index + 1 < len(timestamps)
        else timestamps[end_index] + typical_gap / 2
    )
    if end <= start:
        raise ValueError("derived storyboard proposal has nonpositive duration")
    return max(0.0, start), end


def instructional_plan(
    selection_rows: list[dict[str, Any]],
    storyboard_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    selection = indexed(selection_rows, "item_id", "instructional selection")
    storyboards = indexed(storyboard_rows, "item_id", "instructional storyboards")
    manual = indexed(manual_rows, "item_id", "instructional manual label freeze")
    if set(selection) != set(storyboards) or set(selection) != set(manual):
        raise ValueError("instructional plan inputs do not exactly cover one cohort")
    output = []
    for item_id in sorted(selection):
        gold, source, storyboard = manual[item_id], selection[item_id], storyboards[item_id]
        if gold.get("visual_demo") != "yes":
            continue
        alignment = gold.get("label_alignment")
        if alignment not in {"exact", "partial"}:
            continue
        label = (
            str(source.get("norm") or "").strip()
            if alignment == "exact"
            else str(gold.get("corrected_behavior_label") or "").strip()
        )
        if not label:
            raise ValueError(f"{item_id}: usable demo lacks explicit behavior label")
        start, end = storyboard_bounds(
            [float(value) for value in storyboard.get("sampled_timestamps") or []],
            int(gold["segment_start_frame"]), int(gold["segment_end_frame"]),
        )
        output.append({
            "candidate_id": f"instructional-artifact:{item_id}",
            "item_id": item_id,
            "uid": source["uid"],
            "pillar": "instructional",
            "source_path": source["source_clip"],
            "source_sha256": source["source_clip_sha256"],
            "source_lineage_sealed": True,
            "proposed_start_sec": start,
            "proposed_end_sec": end,
            "bounds_basis": "36_frame_storyboard_proposal_needs_exact_video_audit",
            "behavior_label": label,
            "label_relation": alignment,
            "transform_required": "manual_instructional_audio_policy_required",
            "ready_for_final_render": False,
            "next_required_review": (
                "exact_video_boundary_visible_leakage_and_audio_policy_audit"
            ),
            "approval_status": "unreviewed_localization_candidate",
            "automatic_acceptance": False,
            "corpus_mutation_authorized": False,
        })
    return output


def commentary_plan(
    selected_rows: list[dict[str, Any]],
    blind_rows: list[dict[str, str]],
    post_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    selected = indexed(selected_rows, "window_id", "commentary selection")
    blind = indexed(blind_rows, "window_id", "commentary blind freeze")
    post = indexed(post_rows, "window_id", "commentary label freeze")
    if set(selected) != set(blind) or set(selected) != set(post):
        raise ValueError("commentary plan inputs do not exactly cover one cohort")
    output = []
    for window_id in sorted(selected):
        visual, label, source = blind[window_id], post[window_id], selected[window_id]
        if not (
            visual.get("performed_event_visible") == "yes"
            and visual.get("actor_target_grounded") == "yes"
            and visual.get("before_action_after_complete") == "yes"
            and visual.get("crucial_action_occluded_or_offframe") == "no"
        ):
            continue
        alignment = label.get("label_alignment")
        if alignment not in {"exact", "partial"}:
            continue
        route = label.get("commentary_visual_route")
        if route in {"text_only", "reject"}:
            continue
        if route not in {
            "commentary_visual", "instructional_demo", "relabel_required"
        }:
            raise ValueError(f"{window_id}: unsupported or incomplete visual route")
        behavior = (
            str(source.get("action_label") or "").strip()
            if alignment == "exact"
            else str(label.get("corrected_behavior_label") or "").strip()
        )
        if not behavior:
            raise ValueError(f"{window_id}: visual event lacks explicit behavior label")
        clean_bounds = (
            label.get("start_boundary_clean") == "yes"
            and label.get("end_boundary_clean") == "yes"
        )
        visible_text_clean = visual.get("label_bearing_text_absent") == "yes"
        if route == "commentary_visual" and alignment != "exact":
            raise ValueError(f"{window_id}: commentary visual route requires exact label")
        if route == "relabel_required" and alignment != "partial":
            raise ValueError(f"{window_id}: relabel route requires partial alignment")
        routed_pillar = "instructional" if route == "instructional_demo" else "commentary"
        ready = (
            route == "commentary_visual" and clean_bounds and visible_text_clean
        )
        if route == "instructional_demo":
            transform = "manual_instructional_audio_policy_required"
            next_review = "exact_video_boundary_visible_leakage_and_audio_policy_audit"
        elif route == "relabel_required":
            transform = "manual_commentary_relabel_route_resolution_required"
            next_review = "manual_relabel_and_commentary_visual_route_resolution"
        else:
            transform = (
                "temporal_cut_and_mute"
                if visible_text_clean
                else "temporal_cut_crop_visible_label_text_and_mute"
            )
            next_review = (
                "post_transform_exact_hashed_artifact_audit"
                if ready
                else "manual_crop_or_boundary_repair_plan"
            )
        output.append({
            "candidate_id": f"{routed_pillar}-artifact:{window_id}",
            "window_id": window_id,
            "uid": source["uid"],
            "pillar": routed_pillar,
            "source_pillar": "commentary",
            "manual_route": route,
            "source_path": source["source_path"],
            "source_sha256": source.get("source_sha256"),
            "source_lineage_sealed": bool(source.get("source_sha256")),
            "proposed_start_sec": float(source["window_start_sec"]),
            "proposed_end_sec": float(source["window_end_sec"]),
            "bounds_basis": "manually_reviewed_selected_window",
            "behavior_label": behavior,
            "label_relation": alignment,
            "transform_required": transform,
            "ready_for_final_render": ready,
            "next_required_review": next_review,
            "approval_status": "unreviewed_localization_candidate",
            "automatic_acceptance": False,
            "corpus_mutation_authorized": False,
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="pillar", required=True)
    instruction = subparsers.add_parser("instructional")
    instruction.add_argument("--selection", type=Path, required=True)
    instruction.add_argument("--storyboards", type=Path, required=True)
    instruction.add_argument("--manual", type=Path, required=True)
    instruction.add_argument("--out", type=Path, required=True)
    commentary = subparsers.add_parser("commentary")
    commentary.add_argument("--selected", type=Path, required=True)
    commentary.add_argument("--blind", type=Path, required=True)
    commentary.add_argument("--post", type=Path, required=True)
    commentary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.pillar == "instructional":
        rows = instructional_plan(
            read_jsonl(args.selection), read_jsonl(args.storyboards), read_tsv(args.manual)
        )
    else:
        rows = commentary_plan(
            read_jsonl(args.selected), read_tsv(args.blind), read_tsv(args.post)
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(json.dumps({
        "pillar": args.pillar,
        "localization_candidates": len(rows),
        "ready_for_final_render": sum(row["ready_for_final_render"] for row in rows),
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
