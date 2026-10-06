#!/usr/bin/env python3
"""Select a blind V14 holdout for a response-grounded visual-demo rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from scripts.score_instructional_v9_alignment import strict_visual_event
else:
    from score_instructional_v9_alignment import strict_visual_event


CONTROL_SEED = "20260728-instructional-v14-response-controls-v1"
BLIND_SEED = "20260728-instructional-v14-response-blind-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(seed: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{item_id}".encode()).hexdigest()


def unique_index(path: Path, *, latest_success: bool = False) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if latest_success and (row.get("error") is not None or row.get("result") is None):
            continue
        item_id = str(row["item_id"])
        if item_id in rows and not latest_success:
            raise ValueError(f"{path}: duplicate item_id {item_id}")
        rows[item_id] = row
    return rows


def stage(visual: dict[str, Any], verifier: dict[str, Any]) -> str:
    if not strict_visual_event(visual):
        return "v9a_reject"
    result = verifier["result"]
    if result.get("demo_usable") != "yes":
        return "v10a_reject"
    if result.get("social_response_or_consequence_present") != "yes":
        return "response_reject"
    return "candidate"


def select(
    sources: list[dict[str, Any]],
    storyboards: dict[str, dict[str, Any]],
    visuals: dict[str, dict[str, Any]],
    verifiers: dict[str, dict[str, Any]],
    *,
    controls_per_band: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    item_ids = {str(row["item_id"]) for row in sources}
    if len(item_ids) != len(sources):
        raise ValueError("source contains duplicate item_id values")
    if len({str(row["uid"]) for row in sources}) != len(sources):
        raise ValueError("source must contain one item per UID")
    for name, records in (
        ("storyboard", storyboards),
        ("visual", visuals),
        ("verifier", verifiers),
    ):
        if set(records) != item_ids:
            raise ValueError(
                f"{name} coverage mismatch: missing={len(item_ids - set(records))}, "
                f"extra={len(set(records) - item_ids)}"
            )

    stage_by_item = {
        item_id: stage(visuals[item_id], verifiers[item_id])
        for item_id in item_ids
    }
    candidates = [
        row for row in sources if stage_by_item[str(row["item_id"])] == "candidate"
    ]
    controls: list[tuple[str, dict[str, Any]]] = []
    for band in ("response_reject", "v10a_reject"):
        band_rows = [
            row for row in sources if stage_by_item[str(row["item_id"])] == band
        ]
        band_rows.sort(key=lambda row: rank(CONTROL_SEED, str(row["item_id"])))
        controls.extend((band, row) for row in band_rows[:controls_per_band])

    banded = [
        *(("primary_v14_candidate", row) for row in candidates),
        *controls,
    ]
    random.Random(BLIND_SEED).shuffle(banded)
    semantic: list[dict[str, Any]] = []
    blind: list[dict[str, Any]] = []
    for audit_index, (band, source) in enumerate(banded):
        item_id = str(source["item_id"])
        storyboard = storyboards[item_id]
        candidate_id = f"v14holdout-{audit_index:04d}"
        semantic.append(
            {
                **source,
                "audit_index": audit_index,
                "candidate_id": candidate_id,
                "band": band,
                "candidate_stage": stage_by_item[item_id],
                "storyboard": storyboard,
                "v9a": visuals[item_id],
                "v10a": verifiers[item_id],
            }
        )
        blind.append(
            {
                "audit_index": audit_index,
                "candidate_id": candidate_id,
                "sheet_path": storyboard["sheet_path"],
                "sheet_sha256": storyboard["sheet_sha256"],
            }
        )

    stage_counts = Counter(stage_by_item.values())
    summary = {
        "kind": "instructional_v14_response_holdout_selection",
        "source_rows": len(sources),
        "source_uids": len({str(row["uid"]) for row in sources}),
        "candidate_rows": len(candidates),
        "candidate_uids": len({str(row["uid"]) for row in candidates}),
        "control_rows": len(controls),
        "selected_rows": len(semantic),
        "stage_counts": dict(sorted(stage_counts.items())),
        "control_counts": dict(
            sorted(Counter(band for band, _ in controls).items())
        ),
        "controls_per_band": controls_per_band,
        "control_seed": CONTROL_SEED,
        "blind_seed": BLIND_SEED,
        "semantic_metadata_in_blind_manifest": False,
        "corpus_mutated": False,
    }
    return semantic, blind, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--storyboards", type=Path, required=True)
    parser.add_argument("--v9a", type=Path, required=True)
    parser.add_argument("--v10a", type=Path, required=True)
    parser.add_argument("--controls-per-band", type=int, default=15)
    parser.add_argument("--out-semantic", type=Path, required=True)
    parser.add_argument("--out-blind", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.out_semantic, args.out_blind, args.out_summary):
        if output.exists():
            raise SystemExit(f"refusing to overwrite: {output}")
    if args.controls_per_band < 1:
        raise SystemExit("--controls-per-band must be positive")

    semantic, blind, summary = select(
        read_jsonl(args.source),
        unique_index(args.storyboards),
        unique_index(args.v9a, latest_success=True),
        unique_index(args.v10a, latest_success=True),
        controls_per_band=args.controls_per_band,
    )
    args.out_semantic.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in semantic)
    )
    args.out_blind.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind)
    )
    summary["artifact_sha256"] = {
        "source": sha256(args.source),
        "storyboards": sha256(args.storyboards),
        "v9a": sha256(args.v9a),
        "v10a": sha256(args.v10a),
        "semantic_output": sha256(args.out_semantic),
        "blind_output": sha256(args.out_blind),
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
