#!/usr/bin/env python3
"""Select the frozen simple V13 instructional visual-demo holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

if __package__:
    from scripts.score_instructional_v9_alignment import strict_visual_event
else:
    from score_instructional_v9_alignment import strict_visual_event


SEED = "20260728-instructional-v13-simple-youtube-visual-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_success(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("error") is None and row.get("result") is not None:
            records[str(row["item_id"])] = row
    return records


def unique_index(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item_id = str(row["item_id"])
        if item_id in records:
            raise ValueError(f"{path}: duplicate item_id {item_id}")
        records[item_id] = row
    return records


def excluded_uids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        result.update(str(row["uid"]) for row in read_jsonl(path))
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_candidate(
    source: dict[str, Any],
    visual: dict[str, Any],
    alignment: dict[str, Any],
) -> bool:
    return (
        source.get("source_platform") == "youtube"
        and source.get("polarity") == "violation"
        and strict_visual_event(visual)
        and alignment["result"].get("usable_after_relabel") == "yes"
    )


def select(
    sources: list[dict[str, Any]],
    storyboards: dict[str, dict[str, Any]],
    visuals: dict[str, dict[str, Any]],
    alignments: dict[str, dict[str, Any]],
    exclusions: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {str(row["item_id"]) for row in sources}
    for name, records in (
        ("storyboards", storyboards),
        ("visuals", visuals),
        ("alignments", alignments),
    ):
        if set(records) != expected:
            raise ValueError(
                f"{name} coverage mismatch: {len(records)}/{len(expected)}"
            )

    eligible = [
        row for row in sources if str(row["uid"]) not in exclusions
    ]
    candidates = [
        row
        for row in eligible
        if is_candidate(
            row,
            visuals[str(row["item_id"])],
            alignments[str(row["item_id"])],
        )
    ][:60]
    candidate_ids = {str(row["item_id"]) for row in candidates}

    event_rejects = [
        row
        for row in eligible
        if str(row["item_id"]) not in candidate_ids
        and strict_visual_event(visuals[str(row["item_id"])])
    ]
    non_event_rejects = [
        row
        for row in eligible
        if str(row["item_id"]) not in candidate_ids
        and not strict_visual_event(visuals[str(row["item_id"])])
    ]
    controls: list[tuple[str, dict[str, Any]]] = []
    event_index = 0
    non_event_index = 0
    while len(controls) < 30 and (
        event_index < len(event_rejects) or non_event_index < len(non_event_rejects)
    ):
        if event_index < len(event_rejects):
            controls.append(("control_v9b_reject", event_rejects[event_index]))
            event_index += 1
        if len(controls) == 30:
            break
        if non_event_index < len(non_event_rejects):
            controls.append(("control_v9a_non_event", non_event_rejects[non_event_index]))
            non_event_index += 1

    banded = [
        *(("primary_v13_candidate", row) for row in candidates),
        *controls,
    ]
    random.Random(SEED).shuffle(banded)
    semantic: list[dict[str, Any]] = []
    blind: list[dict[str, Any]] = []
    for audit_index, (band, source) in enumerate(banded):
        item_id = str(source["item_id"])
        storyboard = storyboards[item_id]
        candidate_id = f"v13holdout-{audit_index:04d}"
        semantic.append(
            {
                "audit_index": audit_index,
                "candidate_id": candidate_id,
                "band": band,
                **source,
                "storyboard": storyboard,
                "v9a": visuals[item_id],
                "v9b": alignments[item_id],
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

    summary = {
        "kind": "instructional_v13_simple_youtube_visual_holdout_selection",
        "seed": SEED,
        "source_rows": len(sources),
        "excluded_uids": len(exclusions),
        "eligible_rows": len(eligible),
        "candidate_rows_available": sum(
            is_candidate(
                row,
                visuals[str(row["item_id"])],
                alignments[str(row["item_id"])],
            )
            for row in eligible
        ),
        "candidate_rows_selected": len(candidates),
        "candidate_distinct_uids": len({str(row["uid"]) for row in candidates}),
        "control_rows_selected": len(controls),
        "selected_total": len(semantic),
    }
    return semantic, blind, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--storyboards", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--exclude-selection", type=Path, action="append", default=[])
    parser.add_argument("--semantic-output", type=Path, required=True)
    parser.add_argument("--blind-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    sources = read_jsonl(args.sources)
    storyboards = unique_index(args.storyboards)
    visuals = latest_success(args.visual)
    alignments = latest_success(args.alignment)
    semantic, blind, summary = select(
        sources,
        storyboards,
        visuals,
        alignments,
        excluded_uids(args.exclude_selection),
    )
    args.semantic_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in semantic)
    )
    args.blind_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind)
    )
    summary["artifact_sha256"] = {
        "sources": sha256(args.sources),
        "storyboards": sha256(args.storyboards),
        "visual": sha256(args.visual),
        "alignment": sha256(args.alignment),
        "semantic_output": sha256(args.semantic_output),
        "blind_output": sha256(args.blind_output),
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
