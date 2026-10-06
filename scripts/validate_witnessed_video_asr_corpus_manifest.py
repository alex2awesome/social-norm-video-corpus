#!/usr/bin/env python3
"""Validate corpus witnessed video+ASR media before any VLM scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_media(row: dict[str, Any], manifest_dir: Path) -> Path:
    supplied = Path(str(row.get("candidate_video_path") or ""))
    if supplied.is_absolute():
        return supplied
    candidates = (Path.cwd() / supplied, manifest_dir / supplied)
    return next((path for path in candidates if path.is_file()), candidates[-1])


def validate(
    rows: list[dict[str, Any]],
    manifest_dir: Path,
    *,
    expected_candidates: int,
    hash_sample: int,
    seed: str,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    ids = [str(row.get("candidate_id") or "") for row in rows]
    duplicate_ids = len(ids) - len(set(ids))
    if len(rows) != expected_candidates:
        issues.append({
            "kind": "candidate_count_mismatch",
            "detail": f"expected {expected_candidates}, found {len(rows)}",
        })
    if not all(ids):
        issues.append({"kind": "empty_candidate_id", "detail": "one or more rows"})
    if duplicate_ids:
        issues.append({"kind": "duplicate_candidate_id", "detail": str(duplicate_ids)})

    missing_media = 0
    contract_errors = 0
    timing_errors = 0
    context_before_nonempty = 0
    context_after_nonempty = 0
    platforms = Counter()
    paths: dict[str, Path] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        uid = str(row.get("uid") or "")
        platforms[uid.split("__", 1)[0] if "__" in uid else "unknown"] += 1
        before, after = row.get("context_before"), row.get("context_after")
        if not isinstance(before, list) or not isinstance(after, list):
            contract_errors += 1
        else:
            context_before_nonempty += bool(before)
            context_after_nonempty += bool(after)
        if not (
            row.get("two_stage_full_proxy_then_candidate_cut") is True
            and int(row.get("max_source_frames") or 0) == 96
            and int(row.get("max_width") or 0) == 512
            and 0 < float(row.get("sampling_fps") or 0) <= 3.0
            and row.get("error") in (None, "")
        ):
            contract_errors += 1
        start = float(row.get("candidate_relative_start_sec") or 0)
        end = float(row.get("candidate_relative_end_sec") or 0)
        duration = float(row.get("window_duration_sec") or 0)
        if not (0 <= start <= end <= duration + 0.05 and duration > 0):
            timing_errors += 1
        path = resolve_media(row, manifest_dir)
        paths[candidate_id] = path
        if not path.is_file() or path.stat().st_size <= 0:
            missing_media += 1
    for kind, count in (
        ("representation_contract_error", contract_errors),
        ("candidate_timing_error", timing_errors),
        ("missing_or_empty_media", missing_media),
    ):
        if count:
            issues.append({"kind": kind, "detail": str(count)})

    sample_ids = sorted(
        (candidate_id for candidate_id in set(ids) if candidate_id),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )[:hash_sample]
    hash_mismatches = []
    for candidate_id in sample_ids:
        row = next(row for row in rows if str(row.get("candidate_id")) == candidate_id)
        path = paths[candidate_id]
        if path.is_file() and sha256(path) != row.get("candidate_video_sha256"):
            hash_mismatches.append(candidate_id)
    if hash_mismatches:
        issues.append({
            "kind": "sampled_media_hash_mismatch",
            "detail": ",".join(hash_mismatches[:10]),
        })
    return {
        "kind": "witnessed_video_asr_corpus_manifest_validation_v1",
        "candidate_rows": len(rows),
        "expected_candidates": expected_candidates,
        "unique_candidate_ids": len(set(ids)),
        "unique_items": len({str(row.get("item_id")) for row in rows}),
        "unique_uids": len({str(row.get("uid")) for row in rows}),
        "platform_candidate_rows": dict(sorted(platforms.items())),
        "context_before_nonempty_fraction": context_before_nonempty / len(rows) if rows else None,
        "context_after_nonempty_fraction": context_after_nonempty / len(rows) if rows else None,
        "hash_sample_items": len(sample_ids),
        "hash_sample_seed": seed,
        "issues": issues,
        "passed": not issues,
        "policy": "pre_score_validation_no_corpus_mutation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, required=True)
    parser.add_argument("--hash-sample", type=int, default=256)
    parser.add_argument("--seed", default="witnessed-video-asr-manifest-validation-v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    report = validate(
        read_jsonl(args.manifest),
        args.manifest.parent,
        expected_candidates=args.expected_candidates,
        hash_sample=args.hash_sample,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
