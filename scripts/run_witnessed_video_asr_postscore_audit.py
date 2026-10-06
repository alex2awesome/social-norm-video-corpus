#!/usr/bin/env python3
"""Wait CPU-only for witnessed scoring, then freeze and render its audit sample.

This helper avoids remote polling. It reads append-only score progress at a
bounded interval and runs only deterministic selection plus derived manual
media rendering after exact successful-ID coverage. It never changes corpus
media, metadata, labels, or dispositions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def completion(
    manifest_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]],
    expected_candidates: int,
) -> dict[str, Any]:
    manifest_ids = [str(row.get("candidate_id") or "") for row in manifest_rows]
    if (
        len(manifest_ids) != expected_candidates
        or not all(manifest_ids)
        or len(set(manifest_ids)) != len(manifest_ids)
    ):
        raise ValueError("manifest candidate-ID contract failed")
    successful: set[str] = set()
    attempted: set[str] = set()
    unexpected: set[str] = set()
    manifest_set = set(manifest_ids)
    for row in score_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        attempted.add(candidate_id)
        if candidate_id not in manifest_set:
            unexpected.add(candidate_id)
        elif row.get("error") in (None, "") and isinstance(row.get("result"), dict):
            successful.add(candidate_id)
    return {
        "expected_candidates": expected_candidates,
        "score_rows": len(score_rows),
        "attempted_candidate_ids": len(attempted & manifest_set),
        "successful_candidate_ids": len(successful),
        "remaining_candidate_ids": expected_candidates - len(successful),
        "successful_coverage_fraction": len(successful) / expected_candidates,
        "unexpected_candidate_ids": sorted(unexpected)[:20],
        "ready": len(successful) == expected_candidates and not unexpected,
    }


def process_alive(pid_file: Path) -> bool:
    if not pid_file.is_file():
        return False
    try:
        pid = int(json.loads(pid_file.read_text())["pid"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        raise ValueError("invalid scoring PID file")
    try:
        Path(f"/proc/{pid}").stat()
    except FileNotFoundError:
        return False
    return True


def audit_ready(
    progress: dict[str, Any], *, scoring_alive: bool, minimum_coverage: float
) -> bool:
    return (
        not scoring_alive
        and not progress["unexpected_candidate_ids"]
        and float(progress["successful_coverage_fraction"]) >= minimum_coverage
    )


def write_status(path: Path, state: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "witnessed_video_asr_postscore_audit_status_v1",
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
        **fields,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, required=True)
    parser.add_argument("--scoring-pid-file", type=Path, required=True)
    parser.add_argument("--minimum-coverage", type=float, default=0.98)
    parser.add_argument("--exclude-uids", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--poll-seconds", type=float, default=300)
    parser.add_argument("--timeout-seconds", type=float, default=21600)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.poll_seconds < 10 or args.timeout_seconds <= 0
        or not 0 < args.minimum_coverage <= 1
    ):
        raise SystemExit("invalid polling interval or timeout")
    if args.selection_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.selection_dir}")
    manifest_rows = read_jsonl(args.manifest)
    started = time.monotonic()
    while True:
        progress = completion(
            manifest_rows, read_jsonl(args.scores), args.expected_candidates
        )
        scoring_alive = process_alive(args.scoring_pid_file)
        ready = audit_ready(
            progress, scoring_alive=scoring_alive,
            minimum_coverage=args.minimum_coverage,
        )
        write_status(
            args.status, "ready" if ready else "waiting_for_scores",
            elapsed_seconds=time.monotonic() - started, progress=progress,
            scoring_process_alive=scoring_alive,
            minimum_successful_coverage=args.minimum_coverage,
        )
        if ready:
            break
        if time.monotonic() - started >= args.timeout_seconds:
            write_status(
                args.status, "timed_out", elapsed_seconds=time.monotonic() - started,
                progress=progress, scoring_process_alive=scoring_alive,
                minimum_successful_coverage=args.minimum_coverage,
            )
            return 2
        time.sleep(args.poll_seconds)

    selector = args.repo_root / "scripts/select_witnessed_video_asr_corpus_audit.py"
    subprocess.run([
        sys.executable, str(selector), "--manifest", str(args.manifest),
        "--scores", str(args.scores), "--exclude-uids", str(args.exclude_uids),
        "--uniform-clips", "60", "--positive-enrichment", "30",
        "--near-miss-enrichment", "30", "--model-error-enrichment", "10",
        "--out-dir", str(args.selection_dir),
    ], check=True)
    media = args.selection_dir / "manual_media"
    renderer = args.repo_root / "scripts/materialize_witnessed_corpus_audit_media.py"
    subprocess.run([
        sys.executable, str(renderer), "--selection",
        str(args.selection_dir / "sealed_selection.jsonl"), "--out-dir",
        str(media), "--manifest", str(args.selection_dir / "manual_media_manifest.jsonl"),
        "--failures", str(args.selection_dir / "manual_media_failures.jsonl"),
        "--summary", str(args.selection_dir / "manual_media_summary.json"),
        "--ffmpeg", args.ffmpeg, "--ffprobe", args.ffprobe,
    ], check=True)
    write_status(
        args.status, "audit_media_ready", elapsed_seconds=time.monotonic() - started,
        progress=progress, selection_dir=str(args.selection_dir),
        scoring_process_alive=False,
        minimum_successful_coverage=args.minimum_coverage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
