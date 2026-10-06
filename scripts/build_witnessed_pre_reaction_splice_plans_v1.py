#!/usr/bin/env python3
"""Create exact-boundary review templates for strict witnessed candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts.witnessed_reaction_av_contract import candidate_positive
except ModuleNotFoundError:
    from witnessed_reaction_av_contract import candidate_positive  # type: ignore[no-redef]


FIELDS = (
    "candidate_id", "item_id", "uid", "candidate_video_path",
    "candidate_video_sha256", "window_duration_sec", "reaction_start_hint_sec",
    "trigger_action_label", "action_start_sec", "action_end_sec",
    "reaction_start_sec", "pre_reaction_action_complete",
    "reaction_signal_before_cut", "exact_behavior_label_supported",
    "start_boundary_clean", "end_boundary_clean",
    "audiovisual_boundary_reviewed", "manual_rationale", "ready_for_splice",
)

ATOMIC_FIELDS = (
    "pre_reaction_action_complete", "reaction_signal_before_cut",
    "exact_behavior_label_supported", "start_boundary_clean",
    "end_boundary_clean", "audiovisual_boundary_reviewed",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def index(rows: list[dict[str, Any]], field: str, name: str) -> dict[str, dict[str, Any]]:
    output = {str(row.get(field) or ""): row for row in rows}
    if not output or not all(output) or len(output) != len(rows):
        raise ValueError(f"{name} has missing or duplicate {field}")
    return output


def build(
    model_manifest: list[dict[str, Any]], manual_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    media = index(model_manifest, "candidate_id", "witnessed media manifest")
    manual = index(manual_rows, "candidate_id", "witnessed manual gold")
    if set(media) != set(manual):
        raise ValueError("witnessed media and manual cohorts differ")
    output = []
    for candidate_id in sorted(media):
        gold, source = manual[candidate_id], media[candidate_id]
        if not candidate_positive(
            gold, include_authority=False, require_social=True,
            require_unstaged=True,
        ):
            continue
        row = {field: "" for field in FIELDS}
        row.update({
            "candidate_id": candidate_id,
            "item_id": str(source["item_id"]),
            "uid": str(source["uid"]),
            "candidate_video_path": str(source["candidate_video_path"]),
            "candidate_video_sha256": str(source["candidate_video_sha256"]),
            "window_duration_sec": str(source["window_duration_sec"]),
            "reaction_start_hint_sec": str(source["candidate_relative_start_sec"]),
        })
        output.append(row)
    return output


def validate(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ids = [row.get("candidate_id", "") for row in rows]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("splice ledger has missing or duplicate candidate ids")
    output = []
    for row in rows:
        candidate_id = row["candidate_id"]
        if not row.get("trigger_action_label", "").strip():
            raise ValueError(f"{candidate_id}: missing trigger_action_label")
        try:
            duration = float(row["window_duration_sec"])
            action_start = float(row["action_start_sec"])
            action_end = float(row["action_end_sec"])
            reaction_start = float(row["reaction_start_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{candidate_id}: invalid exact boundaries") from exc
        if not 0 <= action_start < action_end <= reaction_start <= duration:
            raise ValueError(f"{candidate_id}: invalid action/reaction ordering")
        if reaction_start - action_end < 0.10:
            raise ValueError(f"{candidate_id}: less than 100ms reaction safety margin")
        if any(row.get(field) not in {"yes", "no"} for field in ATOMIC_FIELDS):
            raise ValueError(f"{candidate_id}: incomplete atomic boundary audit")
        expected_ready = (
            row.get("pre_reaction_action_complete") == "yes"
            and row.get("reaction_signal_before_cut") == "no"
            and row.get("exact_behavior_label_supported") == "yes"
            and row.get("start_boundary_clean") == "yes"
            and row.get("end_boundary_clean") == "yes"
            and row.get("audiovisual_boundary_reviewed") == "yes"
        )
        if row.get("ready_for_splice") not in {"yes", "no"}:
            raise ValueError(f"{candidate_id}: missing ready_for_splice")
        if (row["ready_for_splice"] == "yes") != expected_ready:
            raise ValueError(f"{candidate_id}: ready_for_splice contradicts evidence")
        if not row.get("manual_rationale", "").strip():
            raise ValueError(f"{candidate_id}: missing manual_rationale")
        output.append({
            "candidate_id": candidate_id,
            "item_id": row["item_id"],
            "uid": row["uid"],
            "pillar": "witnessed",
            "source_path": row["candidate_video_path"],
            "source_sha256": row["candidate_video_sha256"],
            "action_start_sec": action_start,
            "action_end_sec": action_end,
            "reaction_start_sec": reaction_start,
            "behavior_label": row["trigger_action_label"].strip(),
            "ready_for_splice": expected_ready,
            "boundary_review_complete": True,
            "approval_status": "unreviewed_exact_splice_plan",
            "automatic_acceptance": False,
            "corpus_mutation_authorized": False,
        })
    return output


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    template = subparsers.add_parser("template")
    template.add_argument("--model-manifest", type=Path, required=True)
    template.add_argument("--manual", type=Path, required=True)
    template.add_argument("--out", type=Path, required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--ledger", type=Path, required=True)
    seal.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.command == "template":
        rows = build(read_jsonl(args.model_manifest), read_tsv(args.manual))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_tsv(args.out, rows)
        result = {"strict_candidates": len(rows), "boundary_review_complete": False}
    else:
        rows = validate(read_tsv(args.ledger))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        result = {
            "reviewed": len(rows),
            "ready_for_splice": sum(row["ready_for_splice"] for row in rows),
            "boundary_review_complete": True,
        }
    print(json.dumps({
        **result, "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
