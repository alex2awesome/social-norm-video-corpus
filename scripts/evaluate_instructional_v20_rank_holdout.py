#!/usr/bin/env python3
"""Reveal V20 rank bands only after joining the frozen blind visual ledger."""

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


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = sum(bool(row["manual_visual_demo"]) for row in rows)
    return {
        "items": len(rows),
        "visual_demos": positives,
        "visual_demo_ratio": positives / len(rows) if rows else None,
        "forms": dict(
            Counter(
                str(row["visual_form"])
                for row in rows
                if row["manual_visual_demo"]
            )
        ),
        "non_demo_forms": dict(
            Counter(
                str(row["visual_form"])
                for row in rows
                if not row["manual_visual_demo"]
            )
        ),
    }


def evaluate(
    sealed: list[dict[str, Any]],
    ledger: list[dict[str, str]],
    expected_count: int = 60,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(sealed) != expected_count or len(ledger) != expected_count:
        raise ValueError(
            f"expected {expected_count} sealed and {expected_count} blind rows"
        )
    by_index = {int(row["audit_index"]): row for row in ledger}
    if len(by_index) != expected_count:
        raise ValueError("duplicate blind audit index")
    output = []
    for source in sealed:
        index = int(source["audit_index"])
        manual = by_index[index]
        if manual["candidate_id"] != source["candidate_id"]:
            raise ValueError(f"candidate mismatch at {index}")
        output.append(
            {
                **source,
                "manual_visual_demo": manual["visual_demo"].strip().lower()
                in {"y", "yes"},
                "manual_visual_conclusive": (
                    manual["visual_conclusive"] == "yes"
                ),
                "visual_form": manual["visual_form"],
                "blind_visual_note": manual["blind_visual_note"],
            }
        )
    bands = {
        band: metric(
            [row for row in output if row["sealed_rank_band"] == band]
        )
        for band in ("high", "boundary_below", "low")
    }
    return output, {
        "kind": "instructional_v20_rank_holdout_manual_evaluation",
        "status": "source_disjoint_transfer_complete_no_promotion",
        "policy": "read_only_no_keep_reject_or_corpus_mutation",
        "overall": metric(output),
        "bands": bands,
        "high_vs_low_visual_ratio_delta": (
            bands["high"]["visual_demo_ratio"]
            - bands["low"]["visual_demo_ratio"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=60)
    args = parser.parse_args()
    rows, summary = evaluate(
        read_jsonl(args.sealed),
        read_tsv(args.ledger),
        expected_count=args.expected_count,
    )
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary["artifact_sha256"] = {
        "sealed": sha256(args.sealed),
        "blind_ledger": sha256(args.ledger),
        "evaluated": sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
