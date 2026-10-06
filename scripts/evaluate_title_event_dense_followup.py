#!/usr/bin/env python3
"""Evaluate a frozen dense follow-up without overwriting the first-pass audit."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_commentary_title_event_cues import FOLLOWUP, STRICT
else:
    from evaluate_commentary_title_event_cues import FOLLOWUP, STRICT


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def unique(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(row["audit_index"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate audit_index")
    return result


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - half, center + half]


def evaluate(
    post_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    corpus_candidates: int,
) -> dict[str, Any]:
    post = unique(post_rows)
    dense = unique(dense_rows)
    expected = {
        index for index, row in post.items()
        if row["disposition"] in FOLLOWUP
    }
    if set(dense) != expected:
        raise ValueError("dense review must cover every and only first-pass follow-up")
    merged = []
    for index in sorted(post):
        row = dict(post[index])
        if index in dense:
            row["first_pass_disposition"] = row["disposition"]
            row["disposition"] = dense[index]["final_disposition"]
            row["dense_outcome"] = dense[index]["dense_outcome"]
        merged.append(row)

    strict = sum(row["disposition"] in STRICT for row in merged)
    unresolved = sum(row["disposition"] in FOLLOWUP for row in merged)
    interval = wilson(strict, len(merged))
    return {
        "kind": "commentary_title_event_post_localization_evaluation",
        "policy": "shadow_only_no_keep_reject",
        "items": len(merged),
        "first_pass_followups": len(dense),
        "dense_outcomes": dict(Counter(row["dense_outcome"] for row in dense.values())),
        "final_dispositions": dict(Counter(row["disposition"] for row in merged)),
        "strict_usable": strict,
        "strict_rate": strict / len(merged),
        "unresolved": unresolved,
        "unresolved_rate": unresolved / len(merged),
        "strict_rate_wilson_95": interval,
        "corpus_candidates": corpus_candidates,
        "estimated_strict_survivors": corpus_candidates * strict / len(merged),
        "estimated_strict_survivors_wilson_95": [
            corpus_candidates * interval[0],
            corpus_candidates * interval[1],
        ],
        "recovered_audit_indices": [
            index for index, row in dense.items()
            if row["dense_outcome"] == "strict_recovered"
        ],
        "unresolved_audit_indices": [
            index for index, row in dense.items()
            if row["dense_outcome"] == "still_unresolved"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--corpus-candidates", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        load_jsonl(args.post),
        load_jsonl(args.dense),
        args.corpus_candidates,
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
