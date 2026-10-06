#!/usr/bin/env python3
"""Summarize outstanding append-only witnessed VLM failures without raw media."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

if __package__:
    from scripts.run_witnessed_reaction_candidate_av_vlm import successful_candidate_ids
else:
    from run_witnessed_reaction_candidate_av_vlm import successful_candidate_ids


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def error_class(value: Any) -> str:
    text = str(value or "missing_result")
    text = re.sub(r"candidate_id=[^ ]+", "candidate_id=<redacted>", text)
    text = re.sub(r"(?<=[\"'])/[^\"']+(?=[\"'])", "<path>", text)
    return text[:240]


def summarize(
    manifest: list[dict[str, Any]], outputs: list[dict[str, Any]], model: str
) -> dict[str, Any]:
    manifest_ids = [str(row.get("candidate_id") or "") for row in manifest]
    if not manifest_ids or any(not value for value in manifest_ids):
        raise ValueError("manifest contains an empty candidate_id")
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("manifest contains duplicate candidate_id values")
    expected = set(manifest_ids)
    successful = successful_candidate_ids(outputs, model) & expected
    latest: dict[str, dict[str, Any]] = {}
    attempts: Counter[str] = Counter()
    unexpected: set[str] = set()
    for row in outputs:
        if row.get("model") != model:
            continue
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        if candidate_id not in expected:
            unexpected.add(candidate_id)
            continue
        latest[candidate_id] = row
        attempts[candidate_id] += 1
    outstanding = expected - successful
    errors = Counter(
        error_class(latest.get(candidate_id, {}).get("error"))
        for candidate_id in outstanding
    )
    attempt_histogram = Counter(attempts.get(candidate_id, 0) for candidate_id in expected)
    return {
        "kind": "witnessed_video_asr_outstanding_failure_summary_v1",
        "model": model,
        "expected_candidates": len(expected),
        "successful_candidates": len(successful),
        "successful_coverage_fraction": len(successful) / len(expected),
        "outstanding_candidates": len(outstanding),
        "unexpected_candidate_ids": len(unexpected),
        "latest_error_counts": dict(sorted(errors.items())),
        "attempt_count_histogram": {
            str(key): value for key, value in sorted(attempt_histogram.items())
        },
        "policy": "diagnostic_only_append_only_retry_preserves_prior_attempts",
        "corpus_mutated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = summarize(read_jsonl(args.manifest), read_jsonl(args.scores), args.model)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
