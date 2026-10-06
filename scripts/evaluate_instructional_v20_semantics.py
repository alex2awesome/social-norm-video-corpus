#!/usr/bin/env python3
"""Evaluate frozen V20 visual labels against a second-pass semantic ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yes(value: str) -> bool:
    if value not in {"Y", "N"}:
        raise ValueError(f"expected Y/N, got {value!r}")
    return value == "Y"


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    visual = sum(bool(row["manual_visual_demo"]) for row in rows)
    exact = sum(bool(row["manual_exact_original"]) for row in rows)
    usable = sum(bool(row["manual_relabel_usable"]) for row in rows)
    recut = sum(bool(row["manual_needs_recut"]) for row in rows)
    return {
        "items": count,
        "visual_demos": visual,
        "visual_demo_ratio": visual / count if count else None,
        "exact_original_labels": exact,
        "exact_original_ratio": exact / count if count else None,
        "relabel_usable": usable,
        "relabel_usable_ratio": usable / count if count else None,
        "needs_recut": recut,
        "failure_modes": dict(
            Counter(
                str(row["semantic_failure_mode"])
                for row in rows
                if row["semantic_failure_mode"] != "visual_non_demo"
            )
        ),
    }


def evaluate(
    visual_rows: list[dict[str, Any]],
    semantic_ledger: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(visual_rows) != 60:
        raise ValueError("expected 60 frozen visual audit rows")
    visual_positives = [
        row for row in visual_rows if row["manual_visual_demo"]
    ]
    if len(semantic_ledger) != len(visual_positives):
        raise ValueError("semantic ledger must cover every visual positive")
    by_index = {int(row["audit_index"]): row for row in semantic_ledger}
    if len(by_index) != len(semantic_ledger):
        raise ValueError("duplicate semantic audit index")

    output: list[dict[str, Any]] = []
    for source in visual_rows:
        index = int(source["audit_index"])
        manual = by_index.get(index)
        if not source["manual_visual_demo"]:
            if manual is not None:
                raise ValueError(f"semantic row for visual non-demo at {index}")
            semantic = {
                "manual_exact_original": False,
                "manual_relabel_usable": False,
                "manual_needs_recut": False,
                "corrected_event": "",
                "semantic_failure_mode": "visual_non_demo",
                "semantic_note": "",
            }
        else:
            if manual is None:
                raise ValueError(f"missing semantic row for visual demo at {index}")
            if manual["candidate_id"] != source["candidate_id"]:
                raise ValueError(f"candidate mismatch at {index}")
            exact = _yes(manual["exact_original"])
            usable = _yes(manual["relabel_usable"])
            recut = _yes(manual["needs_recut"])
            if exact and not usable:
                raise ValueError(f"exact label cannot be unusable at {index}")
            semantic = {
                "manual_exact_original": exact,
                "manual_relabel_usable": usable,
                "manual_needs_recut": recut,
                "corrected_event": manual["corrected_event"],
                "semantic_failure_mode": manual["failure_mode"],
                "semantic_note": manual["semantic_note"],
            }
        output.append({**source, **semantic})

    bands = {
        band: metric(
            [row for row in output if row["sealed_rank_band"] == band]
        )
        for band in ("high", "boundary_below", "low")
    }
    summary = {
        "kind": "instructional_v20_rank_holdout_semantic_evaluation",
        "status": "source_disjoint_manual_audit_complete_no_promotion",
        "policy": "read_only_no_keep_reject_or_corpus_mutation",
        "definitions": {
            "exact_original": (
                "the enacted audiovisual event supports the original norm and polarity"
            ),
            "relabel_usable": (
                "the clip contains a social event usable after preserving or repairing "
                "its event-level label"
            ),
            "needs_recut": (
                "the useful event exists but substantial unrelated or overlong context "
                "should be removed before training"
            ),
        },
        "overall": metric(output),
        "bands": bands,
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-evaluation", type=Path, required=True)
    parser.add_argument("--semantic-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = evaluate(
        read_jsonl(args.visual_evaluation),
        read_tsv(args.semantic_ledger),
    )
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary["artifact_sha256"] = {
        "visual_evaluation": sha256(args.visual_evaluation),
        "semantic_ledger": sha256(args.semantic_ledger),
        "evaluated": sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
