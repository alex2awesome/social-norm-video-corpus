#!/usr/bin/env python3
"""Filter an append-only witnessed score log to one sealed candidate cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.select_witnessed_video_asr_corpus_audit import successful_scores
except ModuleNotFoundError:
    from select_witnessed_video_asr_corpus_audit import successful_scores  # type: ignore[no-redef]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_ids(rows: list[dict[str, Any]]) -> list[str]:
    ids = [
        str(candidate.get("candidate_id") or "")
        for clip in rows
        for candidate in clip.get("candidates") or []
    ]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("selection has missing or duplicate candidate ids")
    return ids


def filter_scores(
    selection_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ids = selected_ids(selection_rows)
    scores = {
        candidate_id: row
        for candidate_id, row in successful_scores(score_rows).items()
        if row.get("error") in (None, "") and isinstance(row.get("result"), dict)
    }
    missing = sorted(set(ids) - set(scores))
    if missing:
        raise ValueError(f"successful scores missing {len(missing)} selected candidates")
    # Preserve sealed selection order, not append-log order.
    return [scores[candidate_id] for candidate_id in ids]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = filter_scores(read_jsonl(args.selection), read_jsonl(args.scores))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = {
        "kind": "witnessed_scores_exact_selection_slice",
        "candidates": len(rows),
        "successful_coverage": 1.0,
        "selection_order_preserved": True,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
        "output_sha256": sha256(args.out),
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
