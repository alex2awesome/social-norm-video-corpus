#!/usr/bin/env python3
"""Evaluate a frozen commentary title-cue sample without promoting it to a keep rule."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    counts = Counter(str(row[field]) for row in rows)
    known = [row for row in rows if row[field] != "unknown"]
    return {
        "counts": dict(sorted(counts.items())),
        "known_items": len(known),
        "yes_ratio_known": (
            sum(row[field] == "yes" for row in known) / len(known) if known else None
        ),
    }


def evaluate(
    selected: list[dict[str, Any]],
    ledger: list[dict[str, str]],
    expected_count: int = 24,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(selected) != expected_count or len(ledger) != expected_count:
        raise ValueError(f"expected {expected_count} selected and ledger rows")
    by_index = {int(row["audit_index"]): row for row in ledger}
    if len(by_index) != expected_count:
        raise ValueError("duplicate audit index")
    output: list[dict[str, Any]] = []
    for source in selected:
        index = int(source["audit_index"])
        manual = by_index[index]
        if source["candidate_id"] != manual["candidate_id"]:
            raise ValueError(f"candidate mismatch at {index}")
        output.append({**source, **manual, "audit_index": index})

    cue_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        for cue in row.get("title_event_cues", []):
            cue_rows[str(cue)].append(row)
    summary = {
        "kind": "commentary_title_candidate_source_disjoint_manual_evaluation",
        "status": "candidate_localization_transfer_complete_no_promotion",
        "policy": "ranking_and_localization_only_no_keep_reject_or_corpus_mutation",
        "items": len(output),
        "render_status": dict(Counter(row["render_status"] for row in output)),
        "source_has_candidate_event_footage": distribution(
            output, "source_has_candidate_event_footage"
        ),
        "usable_for_visual_localization_review": distribution(
            output, "usable_for_visual_localization_review"
        ),
        "title_action_visible": distribution(output, "title_action_visible"),
        "per_title_cue": {
            cue: {
                "items": len(rows),
                "source_event_footage": distribution(
                    rows, "source_has_candidate_event_footage"
                ),
                "title_action_visible": distribution(rows, "title_action_visible"),
            }
            for cue, rows in sorted(cue_rows.items())
        },
    }
    return output, summary


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=24)
    args = parser.parse_args()
    rows, summary = evaluate(
        read_jsonl(args.selected), read_tsv(args.ledger), args.expected_count
    )
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary["artifact_sha256"] = {
        "selected": sha256(args.selected),
        "ledger": sha256(args.ledger),
        "evaluated": sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
