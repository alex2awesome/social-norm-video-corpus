#!/usr/bin/env python3
"""Freeze a source-disjoint holdout for retro provenance plus duration."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(seed: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}\0{row['item_id']}".encode()).hexdigest()


def cohort_name(row: dict[str, Any], source: str, minimum: float) -> str:
    retro = str(row.get("query_source") or "") == source
    duration = float(row.get("duration_hint") or 0)
    if retro and duration >= minimum:
        return "signal_positive"
    if retro:
        return "short_retro_boundary"
    if duration >= minimum:
        return "duration_matched_nonretro_control"
    return "ineligible"


def cohort_availability(
    rows: list[dict[str, Any]],
    excluded_uids: set[str],
    *,
    source: str = "retro_instr_scan",
    minimum_duration: float = 5.0,
) -> dict[str, Any]:
    cohorts = (
        "signal_positive", "short_retro_boundary",
        "duration_matched_nonretro_control",
    )
    all_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in cohorts}
    eligible: dict[str, list[dict[str, Any]]] = {name: [] for name in cohorts}
    for row in rows:
        name = cohort_name(row, source, minimum_duration)
        if name not in all_rows or not row.get("uid") or not row.get("source_clip"):
            continue
        all_rows[name].append(row)
        if str(row["uid"]) not in excluded_uids:
            eligible[name].append(row)

    def counts(values: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "rows": len(values),
            "source_uids": len({str(row["uid"]) for row in values}),
        }

    return {
        "kind": "instructional_retro_duration_holdout_availability_v1",
        "source_rows": len(rows),
        "excluded_source_uids": len(excluded_uids),
        "signal": {
            "query_source": source,
            "minimum_duration_seconds": minimum_duration,
        },
        "before_exclusion": {
            name: counts(all_rows[name]) for name in cohorts
        },
        "after_exclusion": {
            name: counts(eligible[name]) for name in cohorts
        },
        "policy": "diagnostic_only_no_selection_or_corpus_mutation",
    }


def select(
    rows: list[dict[str, Any]],
    excluded_uids: set[str],
    counts: dict[str, int],
    *,
    seed: str,
    source: str = "retro_instr_scan",
    minimum_duration: float = 5.0,
) -> list[dict[str, Any]]:
    if not counts or any(value < 0 for value in counts.values()):
        raise ValueError("cohort counts must be nonnegative")
    expected_cohorts = {
        "signal_positive", "short_retro_boundary", "duration_matched_nonretro_control"
    }
    if set(counts) != expected_cohorts:
        raise ValueError("cohort counts do not exactly match the frozen design")
    seen_items: set[str] = set()
    pools = {name: [] for name in expected_cohorts}
    for row in rows:
        item_id, uid = str(row.get("item_id") or ""), str(row.get("uid") or "")
        if not item_id or not uid or not row.get("source_clip"):
            continue
        if item_id in seen_items:
            raise ValueError(f"duplicate item_id: {item_id}")
        seen_items.add(item_id)
        name = cohort_name(row, source, minimum_duration)
        if uid not in excluded_uids and name in pools:
            pools[name].append(row)
    selected: list[dict[str, Any]] = []
    used_uids: set[str] = set()
    # Allocate scarce provenance cohorts before the abundant control pool.
    # The order is part of the frozen sampling contract: a UID containing both
    # retro and ordinary demos must not be consumed as a control first.
    allocation_order = (
        "short_retro_boundary",
        "signal_positive",
        "duration_matched_nonretro_control",
    )
    for name in allocation_order:
        pool = sorted(pools[name], key=lambda row: stable_key(f"{seed}:{name}", row))
        chosen = []
        for row in pool:
            uid = str(row["uid"])
            if uid in used_uids:
                continue
            chosen.append({**row, "holdout_cohort": name})
            used_uids.add(uid)
            if len(chosen) == counts[name]:
                break
        if len(chosen) != counts[name]:
            raise ValueError(f"insufficient source-disjoint rows for {name}")
        selected.extend(chosen)
    return sorted(selected, key=lambda row: stable_key(f"{seed}:blind", row))


def blind_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "audit_index": index,
        "candidate_id": f"retro-duration-holdout-{index:04d}",
        "item_id": str(row["item_id"]),
        "uid": str(row["uid"]),
        "pillar": "instructional",
        "source_clip": str(row["source_clip"]),
    }


BLIND_FIELDS = (
    "audit_index", "candidate_id", "item_id", "uid", "visual_form",
    "visual_demo", "complete_demo", "participant_grounding",
    "blind_evidence", "blind_note",
)
POST_FIELDS = (
    "audit_index", "candidate_id", "item_id", "label_is_concrete_social_behavior",
    "visual_matches_original_label", "usable_after_relabel", "visible_polarity",
    "corrected_behavior", "failure_mode", "post_reveal_evidence", "post_reveal_note",
)


def write_template(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path, action="append", required=True)
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--signal-positive", type=int, default=30)
    parser.add_argument("--short-retro", type=int, default=15)
    parser.add_argument("--nonretro-control", type=int, default=15)
    parser.add_argument("--seed", default="instructional-retro-duration-holdout-v1")
    parser.add_argument("--availability-out", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    excluded = {
        line.strip()
        for path in args.exclude_uids
        for line in path.read_text().splitlines()
        if line.strip()
    }
    excluded.update(
        str(row["uid"])
        for path in args.exclude_jsonl
        for row in read_jsonl(path)
        if row.get("uid")
    )
    counts = {
        "signal_positive": args.signal_positive,
        "short_retro_boundary": args.short_retro,
        "duration_matched_nonretro_control": args.nonretro_control,
    }
    source_rows = read_jsonl(args.source)
    availability = cohort_availability(source_rows, excluded)
    availability["requested_cohorts"] = counts
    availability["individually_sufficient_source_uids"] = {
        name: availability["after_exclusion"][name]["source_uids"] >= requested
        for name, requested in counts.items()
    }
    if args.availability_out:
        if args.availability_out.exists():
            raise SystemExit(f"refusing to overwrite: {args.availability_out}")
        args.availability_out.parent.mkdir(parents=True, exist_ok=True)
        args.availability_out.write_text(
            json.dumps(availability, indent=2, sort_keys=True) + "\n"
        )
    chosen = select(source_rows, excluded, counts, seed=args.seed)
    args.out.mkdir(parents=True)
    sealed, blind = [], []
    for index, row in enumerate(chosen):
        identity = blind_record(row, index)
        sealed.append({**row, **identity})
        blind.append(identity)
    sealed_path, blind_path = args.out / "sealed_selection.jsonl", args.out / "blind_source_manifest.jsonl"
    sealed_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed))
    blind_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in blind))
    write_template(args.out / "manual_blind_visual.tsv", blind, BLIND_FIELDS)
    write_template(args.out / "manual_post_reveal.tsv", blind, POST_FIELDS)
    prereg = {
        "kind": "instructional_retro_duration_independent_holdout_v1",
        "status": "preregistered_blind_manual_audit_pending",
        "signal": {"query_source": "retro_instr_scan", "minimum_duration_seconds": 5.0},
        "items": len(chosen),
        "source_disjoint": len({row["uid"] for row in chosen}) == len(chosen),
        "cohorts": dict(sorted(Counter(row["holdout_cohort"] for row in chosen).items())),
        "manual_review_required_for_every_item": True,
        "gate": {
            "minimum_signal_positive_items": 30,
            "minimum_visual_demo_precision": 0.7,
            "minimum_visual_demo_wilson_95_lower": 0.5,
            "permitted_if_passed": "manual_review_priority_only",
            "automatic_acceptance_allowed": False,
        },
        "policy": "shadow_only_no_keep_reject_delete_or_corpus_mutation",
        "artifact_sha256": {
            "source": sha256(args.source),
            "exclude_uid_files": {str(path): sha256(path) for path in args.exclude_uids},
            "exclude_jsonl_files": {str(path): sha256(path) for path in args.exclude_jsonl},
            "sealed_selection": sha256(sealed_path),
            "blind_source_manifest": sha256(blind_path),
        },
    }
    (args.out / "preregistration.json").write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(prereg, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
