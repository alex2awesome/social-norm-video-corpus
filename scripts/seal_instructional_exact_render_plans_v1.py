#!/usr/bin/env python3
"""Manually seal exact instructional bounds and an evidence-preserving audio policy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "candidate_id", "pillar", "uid", "proposed_start_sec", "proposed_end_sec",
    "exact_start_sec", "exact_end_sec", "visual_event_complete",
    "actor_target_grounded", "exact_behavior_label_supported",
    "start_boundary_clean", "end_boundary_clean",
    "label_bearing_visible_text_absent", "audio_content",
    "audio_is_required_for_behavior", "label_bearing_explanation_audio_overlaps",
    "audio_action", "manual_description", "manual_rationale",
    "ready_for_final_render",
)

YES_NO = {"yes", "no"}
AUDIO_CONTENT = {
    "diegetic_behavior", "nonlabel_background", "label_bearing_explanation",
    "mixed_diegetic_and_explanation", "no_audio", "uncertain",
}
AUDIO_ACTION = {"preserve", "mute", "reject"}
VISUAL_GATES = (
    "visual_event_complete", "actor_target_grounded",
    "exact_behavior_label_supported", "start_boundary_clean",
    "end_boundary_clean", "label_bearing_visible_text_absent",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def indexed(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output = {str(row.get("candidate_id") or ""): row for row in rows}
    if not output or not all(output) or len(output) != len(rows):
        raise ValueError(f"{name} has missing or duplicate candidate_id")
    return output


def template(plans: list[dict[str, Any]]) -> list[dict[str, str]]:
    indexed(plans, "instructional plans")
    output = []
    for row in plans:
        if (
            row.get("pillar") != "instructional"
            or row.get("transform_required") != "manual_instructional_audio_policy_required"
            or row.get("ready_for_final_render") is not False
            or row.get("approval_status") != "unreviewed_localization_candidate"
        ):
            raise ValueError(f"{row.get('candidate_id')}: invalid instructional plan state")
        value = {field: "" for field in FIELDS}
        value.update({
            "candidate_id": str(row["candidate_id"]),
            "pillar": "instructional",
            "uid": str(row["uid"]),
            "proposed_start_sec": str(row["proposed_start_sec"]),
            "proposed_end_sec": str(row["proposed_end_sec"]),
        })
        output.append(value)
    return output


def seal(
    plans: list[dict[str, Any]], reviews: list[dict[str, str]],
) -> list[dict[str, Any]]:
    plan = indexed(plans, "instructional plans")
    review = indexed(reviews, "instructional exact reviews")
    if set(plan) != set(review):
        raise ValueError("exact instructional review does not exactly cover plans")
    output = []
    for candidate_id in sorted(plan):
        source, row = plan[candidate_id], review[candidate_id]
        if set(row) != set(FIELDS):
            raise ValueError(f"{candidate_id}: exact review schema mismatch")
        if row.get("pillar") != "instructional" or row.get("uid") != str(source["uid"]):
            raise ValueError(f"{candidate_id}: exact review lineage mismatch")
        if (
            row.get("proposed_start_sec") != str(source["proposed_start_sec"])
            or row.get("proposed_end_sec") != str(source["proposed_end_sec"])
        ):
            raise ValueError(f"{candidate_id}: proposed-bound lineage changed")
        try:
            start, end = float(row["exact_start_sec"]), float(row["exact_end_sec"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{candidate_id}: invalid exact bounds") from exc
        if not 0 <= start < end:
            raise ValueError(f"{candidate_id}: nonpositive exact interval")
        if any(row.get(field) not in YES_NO for field in VISUAL_GATES):
            raise ValueError(f"{candidate_id}: incomplete visual/boundary audit")
        if row.get("audio_content") not in AUDIO_CONTENT:
            raise ValueError(f"{candidate_id}: invalid audio_content")
        if row.get("audio_is_required_for_behavior") not in YES_NO:
            raise ValueError(f"{candidate_id}: missing behavior-audio dependency")
        if row.get("label_bearing_explanation_audio_overlaps") not in YES_NO:
            raise ValueError(f"{candidate_id}: missing explanation-audio judgment")
        if row.get("audio_action") not in AUDIO_ACTION:
            raise ValueError(f"{candidate_id}: invalid audio_action")
        if not row.get("manual_description", "").strip() or not row.get("manual_rationale", "").strip():
            raise ValueError(f"{candidate_id}: missing manual evidence")

        action = row["audio_action"]
        audio_content = row["audio_content"]
        required = row["audio_is_required_for_behavior"] == "yes"
        leakage = row["label_bearing_explanation_audio_overlaps"] == "yes"
        if audio_content == "uncertain" and action != "reject":
            raise ValueError(f"{candidate_id}: uncertain audio must reject")
        label_audio_kinds = {
            "label_bearing_explanation", "mixed_diegetic_and_explanation"
        }
        if (audio_content in label_audio_kinds) != leakage:
            raise ValueError(
                f"{candidate_id}: audio_content and explanation leakage contradict"
            )
        if required and action == "mute":
            raise ValueError(f"{candidate_id}: cannot mute behavior-required audio")
        if leakage and action == "preserve":
            raise ValueError(f"{candidate_id}: cannot preserve label-bearing explanation")
        if audio_content in label_audio_kinds and action == "preserve":
            raise ValueError(f"{candidate_id}: label-bearing audio cannot be preserved")
        if action == "mute" and audio_content not in label_audio_kinds:
            raise ValueError(
                f"{candidate_id}: clean/nonexistent audio must not be muted"
            )
        if audio_content == "no_audio" and required:
            raise ValueError(f"{candidate_id}: absent audio cannot be behavior-required")

        visual_ready = all(row[field] == "yes" for field in VISUAL_GATES)
        audio_ready = action != "reject" and not (required and action == "mute") and not (leakage and action == "preserve")
        ready = visual_ready and audio_ready
        if row.get("ready_for_final_render") not in YES_NO or (
            row["ready_for_final_render"] == "yes"
        ) != ready:
            raise ValueError(f"{candidate_id}: ready_for_final_render contradicts audit")

        value = dict(source)
        value.update({
            "proposed_start_sec": start,
            "proposed_end_sec": end,
            "bounds_basis": "manual_exact_video_boundary_audit_v1",
            "transform_required": (
                "temporal_cut_preserve_audio" if action == "preserve"
                else "temporal_cut_and_mute" if action == "mute"
                else "rejected_inseparable_or_uncertain_audio"
            ),
            "audio_policy": action,
            "audio_content": audio_content,
            "audio_is_required_for_behavior": required,
            "label_bearing_explanation_audio_overlaps": leakage,
            "ready_for_final_render": ready,
            "exact_boundary_and_audio_policy_review_complete": True,
            "automatic_acceptance": False,
            "corpus_mutation_authorized": False,
        })
        output.append(value)
    return output


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("template")
    make.add_argument("--plan", type=Path, required=True)
    make.add_argument("--out", type=Path, required=True)
    finish = sub.add_parser("seal")
    finish.add_argument("--plan", type=Path, required=True)
    finish.add_argument("--review", type=Path, required=True)
    finish.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    plans = read_jsonl(args.plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "template":
        rows = template(plans); write_tsv(args.out, rows)
        result = {"plans": len(rows), "manual_review_complete": False}
    else:
        rows = seal(plans, read_tsv(args.review))
        args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        result = {
            "reviewed": len(rows),
            "ready_for_final_render": sum(row["ready_for_final_render"] for row in rows),
            "manual_review_complete": True,
        }
    print(json.dumps({
        **result, "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
