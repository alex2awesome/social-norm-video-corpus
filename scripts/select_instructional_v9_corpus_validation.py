#!/usr/bin/env python3
"""Freeze fresh instructional V9 validation and blind-review manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from scripts.score_instructional_v9_alignment import strict_visual_event
else:
    from score_instructional_v9_alignment import strict_visual_event


SEED = "20260728-v9-fresh"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def latest_success(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("result") is None or row.get("error") is not None:
            continue
        records[str(row["item_id"])] = row
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(item_id: str, namespace: str) -> str:
    return hashlib.sha256(f"{SEED}|{namespace}|{item_id}".encode()).hexdigest()


def positive(row: dict[str, Any]) -> bool:
    return (
        row["alignment"]["result"]["usable_after_relabel"] == "yes"
        and strict_visual_event(row["visual"])
    )


def one_per_uid(
    rows: list[dict[str, Any]],
    count: int,
    namespace: str,
    used_uids: set[str],
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: stable_key(row["source"]["item_id"], namespace),
    )
    selected = []
    for row in ordered:
        uid = str(row["source"]["uid"])
        if uid in used_uids:
            continue
        used_uids.add(uid)
        selected.append(row)
        if len(selected) == count:
            break
    return selected


def balanced_rejects(
    rows: list[dict[str, Any]],
    count: int,
    used_uids: set[str],
) -> list[dict[str, Any]]:
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        event = row["visual"]["result"]["observable_event"]
        cells["v9a_event" if event == "yes" else "v9a_no_event"].append(row)
    for name in cells:
        cells[name].sort(
            key=lambda row: stable_key(row["source"]["item_id"], name)
        )
    selected = []
    names = ["v9a_event", "v9a_no_event"]
    while len(selected) < count and any(cells.values()):
        progressed = False
        for name in names:
            while cells[name]:
                row = cells[name].pop(0)
                uid = str(row["source"]["uid"])
                if uid in used_uids:
                    continue
                used_uids.add(uid)
                selected.append(row)
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            break
    return selected


def select(
    manifest: list[dict[str, Any]],
    visual: dict[str, dict[str, Any]],
    alignments: dict[str, dict[str, Any]],
    storyboards: dict[str, dict[str, Any]],
    exclusions: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {str(row["item_id"]) for row in manifest}
    for name, records in (
        ("visual", visual),
        ("alignment", alignments),
        ("storyboard", storyboards),
    ):
        if set(records) != expected:
            raise ValueError(
                f"{name} coverage mismatch: {len(records)}/{len(expected)}"
            )
    joined = [
        {
            "source": row,
            "visual": visual[str(row["item_id"])],
            "alignment": alignments[str(row["item_id"])],
            "storyboard": storyboards[str(row["item_id"])],
        }
        for row in manifest
        if str(row["uid"]) not in exclusions
    ]
    primary = [
        row
        for row in joined
        if row["source"].get("polarity") == "violation" and positive(row)
    ]
    primary.sort(
        key=lambda row: stable_key(row["source"]["item_id"], "primary_all")
    )
    used_uids = {str(row["source"]["uid"]) for row in primary}
    correct = one_per_uid(
        [
            row
            for row in joined
            if row["source"].get("polarity") == "correct" and positive(row)
        ],
        30,
        "correct",
        used_uids,
    )
    explanation = one_per_uid(
        [
            row
            for row in joined
            if row["source"].get("polarity") == "explanation" and positive(row)
        ],
        30,
        "explanation",
        used_uids,
    )
    rejects = balanced_rejects(
        [
            row
            for row in joined
            if row["source"].get("polarity") == "violation" and not positive(row)
        ],
        30,
        used_uids,
    )
    banded = [
        *(("primary_violation_all", row) for row in primary),
        *(("comparison_correct", row) for row in correct),
        *(("comparison_explanation", row) for row in explanation),
        *(("control_violation_reject", row) for row in rejects),
    ]
    rng = random.Random(SEED)
    rng.shuffle(banded)
    output = []
    for audit_index, (band, row) in enumerate(banded):
        source = row["source"]
        output.append(
            {
                "audit_index": audit_index,
                "candidate_id": f"v9fresh-{audit_index:04d}",
                "band": band,
                "item_id": source["item_id"],
                "uid": source["uid"],
                "pillar": source.get("pillar") or "instructional",
                "category": source.get("category"),
                "polarity": source.get("polarity"),
                "norm": source.get("norm"),
                "title": source.get("title"),
                "start_quote": source.get("start_quote"),
                "end_quote": source.get("end_quote"),
                "explanation": source.get("explanation"),
                "source_platform": source.get("source_platform"),
                "storyboard_path": row["storyboard"]["sheet_path"],
                "storyboard_sha256": row["storyboard"]["sheet_sha256"],
                "v9a": row["visual"],
                "v9b": row["alignment"],
            }
        )
    summary = {
        "kind": "instructional_v9_fresh_corpus_validation_selection",
        "seed": SEED,
        "eligible_after_exclusions": len(joined),
        "selected_total": len(output),
        "bands": {
            "primary_violation_all": len(primary),
            "comparison_correct": len(correct),
            "comparison_explanation": len(explanation),
            "control_violation_reject": len(rejects),
        },
        "primary_distinct_uids": len(
            {str(row["source"]["uid"]) for row in primary}
        ),
        "all_primary_outputs_selected": True,
        "corpus_mutated": False,
    }
    return output, summary


def blind_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "audit_index": row["audit_index"],
            "candidate_id": row["candidate_id"],
            "item_id": row["item_id"],
            "uid": row["uid"],
            "pillar": row["pillar"],
            "sheet_path": row["storyboard_path"],
            "sheet_sha256": row["storyboard_sha256"],
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--storyboards", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blind-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    for target in (args.out, args.blind_out, args.summary_out):
        if target.exists():
            raise SystemExit(f"refusing to overwrite: {target}")
    manifest = load_jsonl(args.manifest)
    visual = latest_success(args.visual)
    alignment = latest_success(args.alignment)
    storyboards = {
        str(row["item_id"]): row for row in load_jsonl(args.storyboards)
    }
    exclusions = {
        line.strip()
        for line in args.exclusions.read_text().splitlines()
        if line.strip()
    }
    rows, summary = select(
        manifest, visual, alignment, storyboards, exclusions
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path, records in (
        (args.out, rows),
        (args.blind_out, blind_rows(rows)),
    ):
        with path.open("w") as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary["input_hashes"] = {
        "manifest": sha256(args.manifest),
        "visual": sha256(args.visual),
        "alignment": sha256(args.alignment),
        "storyboards": sha256(args.storyboards),
        "exclusions": sha256(args.exclusions),
    }
    summary["selection_sha256"] = sha256(args.out)
    summary["blind_selection_sha256"] = sha256(args.blind_out)
    args.summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
