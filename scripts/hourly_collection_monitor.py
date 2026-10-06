#!/usr/bin/env python3
"""Record an append-only collection-health snapshot and fresh review queue.

This monitor is deliberately observational.  It never restarts jobs, downloads
media, changes labels, or promotes audit candidates.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov"}
SOURCES = ("dailymotion", "rumble", "youtube", "reddit")
PIPELINE_BACKLOG_ALERTS = {
    "pending_detect": 100,
    "pending_transcribe": 500,
}
WITNESSED_VIDEO_ASR_RUN = "20260806_witnessed_video_asr_v3"
STRICT_AUDIT_COVERAGE_RUN = "20260811_strict_audit_backfill_v1c"
STRICT_AUDIT_QUEUE_RUN = "20260811_strict_audit_backfill_queues_v1"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def process_count(pattern: str) -> int:
    result = subprocess.run(
        ["pgrep", "-fc", pattern], capture_output=True, text=True, check=False
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def source_inventory(discussion: Path, videos: Path) -> dict[str, dict[str, int]]:
    result = {}
    for source in SOURCES:
        records = sum(1 for _ in discussion.glob(f"{source}__*.json"))
        retained = {
            path.stem
            for path in videos.glob(f"{source}__*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_SUFFIXES
            and path.stat().st_size > 0
        }
        result[source] = {
            "records": records,
            "retained": len(retained),
            "missing": max(records - len(retained), 0),
        }
    return result


def ledger_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = row.get("uid")
        if uid:
            latest[str(uid)] = row
    statuses = Counter(row.get("status") for row in latest.values())
    return {
        "events": len(rows),
        "unique_downloaded": statuses["downloaded"],
        "latest_failed": statuses["failed"],
        "auth_failures_total": sum(bool(row.get("auth_failure")) for row in rows),
        "last_event_at": rows[-1].get("at") if rows else None,
        "last_event_status": rows[-1].get("status") if rows else None,
    }


def backfill_gap_status(
    discussion: Path,
    videos: Path,
    ledger: Path,
    sources: tuple[str, ...] = ("dailymotion", "rumble"),
) -> dict[str, dict[str, int]]:
    """Separate never-attempted gaps from latest terminal download failures."""
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(ledger):
        if row.get("uid"):
            latest[str(row["uid"])] = row
    result: dict[str, dict[str, int]] = {}
    for source in sources:
        records = {path.stem for path in discussion.glob(f"{source}__*.json")}
        retained = {
            path.stem
            for path in videos.glob(f"{source}__*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_SUFFIXES
            and path.stat().st_size > 0
        }
        missing = records - retained
        latest_failed = sum(
            latest.get(uid, {}).get("status") == "failed" for uid in missing
        )
        unattempted = sum(uid not in latest for uid in missing)
        nonterminal = len(missing) - latest_failed - unattempted
        result[source] = {
            "missing": len(missing),
            "latest_failed": latest_failed,
            "unattempted": unattempted,
            "other_latest_status": nonterminal,
        }
    return result


def state_status_counts(path: Path) -> dict[str, int]:
    """Return the current crawler/pipeline queue counts without mutating state."""
    if not path.is_file():
        return {}
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT status, COUNT(*) FROM seen_videos GROUP BY status"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return {}
    return {str(status): int(count) for status, count in rows}


def audited_search_status(
    path: Path,
    query_source: str = "manual_audit_search_v1",
) -> dict[str, Any]:
    """Observe the narrow manually-supported retrieval promotion."""
    if not path.is_file():
        return {}
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            query_rows = [
                dict(row)
                for row in connection.execute(
                    """SELECT query, category, priority, active,
                              videos_processed, videos_with_hits,
                              total_reaction_count, last_used
                       FROM queries WHERE source=?
                       ORDER BY category, query""",
                    (query_source,),
                )
            ]
            outcome_rows = [
                dict(row)
                for row in connection.execute(
                    """SELECT status, COALESCE(modality, 'unrouted') AS modality,
                              COUNT(*) AS items
                       FROM seen_videos WHERE query_source=?
                       GROUP BY status, COALESCE(modality, 'unrouted')
                       ORDER BY status, modality""",
                    (query_source,),
                )
            ]
            recent = [
                dict(row)
                for row in connection.execute(
                    """SELECT video_id AS uid, title, query, category, status,
                              modality, enumerated_at, processed_at
                       FROM seen_videos WHERE query_source=?
                       ORDER BY enumerated_at DESC LIMIT 8""",
                    (query_source,),
                )
            ]
        finally:
            connection.close()
    except sqlite3.Error:
        return {}
    return {
        "query_source": query_source,
        "queries": query_rows,
        "active_queries": sum(bool(row["active"]) for row in query_rows),
        "videos_processed": sum(int(row["videos_processed"]) for row in query_rows),
        "videos_with_hits": sum(int(row["videos_with_hits"]) for row in query_rows),
        "outcomes": outcome_rows,
        "recent_candidates": recent,
        "policy": "retrieval_only_requires_manual_output_audit",
    }


def count_lines(path: Path) -> int:
    """Count append-only records without loading a potentially large file."""
    if not path.is_file():
        return 0
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def score_item_status(
    paths: list[Path],
    required_fields: tuple[str, ...] = ("low_level",),
) -> dict[str, bool]:
    """Deduplicate base/shard attempts; any successful retry wins."""
    result: dict[str, bool] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or not row.get("item_id"):
                    continue
                item_id = str(row["item_id"])
                successful = (
                    row.get("error") is None
                    and all(bool(row.get(field)) for field in required_fields)
                )
                result[item_id] = result.get(item_id, False) or successful
    return result


def successful_item_ids(paths: list[Path]) -> set[str]:
    """Return successful ids for callers that only need the resumable subset."""
    return {
        item_id
        for item_id, successful in score_item_status(paths).items()
        if successful
    }


def shadow_score_status(
    data: Path,
    process_counter=process_count,
) -> dict[str, Any] | None:
    """Report the newest immutable full-corpus shadow-scoring run."""
    root = data / "shadow_scores"
    runs = sorted(
        path
        for path in root.glob("*_full_corpus_v*")
        if path.is_dir()
    )
    if not runs:
        return None
    run = runs[-1]
    pillars = {}
    for pillar in ("instructional", "witnessed", "commentary"):
        manifest = run / f"{pillar}_manifest.jsonl"
        output_candidates = [
            path
            for path in run.glob(f"{pillar}_low_level*.jsonl")
            if "canary" not in path.name
        ]
        # The first commentary pass used repeated random H.264 seeks and was
        # frozen as evidence after decoder failures. Once the sequential
        # replacement exists, it is explicitly superseded for progress
        # accounting (the file itself remains untouched).
        if (
            pillar == "commentary"
            and run.joinpath("commentary_low_level_v2_sequential.jsonl").is_file()
        ):
            output_candidates = [
                path
                for path in output_candidates
                if path.name != "commentary_low_level.jsonl"
            ]
        total = count_lines(manifest)
        item_status = score_item_status(output_candidates)
        attempts = len(item_status)
        successful = sum(item_status.values())
        failed = attempts - successful
        running = process_counter(
            rf"[s]core_visual_scene_baselines.py.*{pillar}_manifest.jsonl"
        )
        pillars[pillar] = {
            "manifest_items": total,
            "attempted_items": attempts,
            "successfully_scored_items": successful,
            "failed_or_missing_items": failed,
            "outputs": sorted(path.name for path in output_candidates),
            "remaining_items": max(total - attempts, 0),
            "progress_fraction": attempts / total if total else None,
            "processes": running,
            "complete": bool(total and attempts >= total),
        }
    multimodal_root = run / "multimodal_v1"
    multimodal = None
    if multimodal_root.is_dir():
        multimodal = {}
        for pillar in ("instructional", "witnessed"):
            manifest = run / f"{pillar}_manifest.jsonl"
            output = multimodal_root / f"{pillar}_pose_clip.jsonl"
            total = count_lines(manifest)
            item_status = score_item_status(
                [output],
                required_fields=("low_level", "keypoints", "clip_scores"),
            )
            attempted = len(item_status)
            successful = sum(item_status.values())
            multimodal[pillar] = {
                "manifest_items": total,
                "attempted_items": attempted,
                "successfully_scored_items": successful,
                "failed_items": attempted - successful,
                "remaining_items": max(total - attempted, 0),
                "progress_fraction": attempted / total if total else None,
                "processes": process_counter(
                    rf"[s]core_visual_scene_baselines.py.*{pillar}_manifest.jsonl"
                    rf".*{pillar}_pose_clip.jsonl"
                ),
                "complete": bool(total and attempted >= total),
                "policy": "ranking_only_no_keep_reject",
            }
    return {
        "run": run.name,
        "path": str(run),
        "pillars": pillars,
        "multimodal_v1": multimodal,
        "policy": "shadow_only_non_destructive",
    }


def strict_audit_backfill_status(
    data: Path,
    process_counter=process_count,
    coverage_run: str = STRICT_AUDIT_COVERAGE_RUN,
    queue_run: str = STRICT_AUDIT_QUEUE_RUN,
) -> dict[str, Any] | None:
    """Observe the current corpus-wide strict-audit coverage backfill."""
    coverage = data / "shadow_scores" / coverage_run
    queues = data / "shadow_scores" / queue_run
    if not coverage.is_dir() or not queues.is_dir():
        return None
    coverage_summary = read_json(coverage / "summary.json")
    queue_summary = read_json(queues / "summary.json")
    expected = int(queue_summary.get("low_level_delta_items") or 0)
    outputs = sorted(queues.glob("low_level_shard_*_of_*.jsonl"))
    states = score_item_status(outputs)
    attempted = len(states)
    successful = sum(states.values())
    processes = process_counter(
        rf"[s]core_visual_scene_baselines.py.*{queue_run}/low_level_delta_manifest.jsonl"
    )
    complete = bool(expected and attempted >= expected)
    return {
        "coverage_run": coverage_run,
        "queue_run": queue_run,
        "positive_media_items": sum(
            int(value)
            for pillar, value in (coverage_summary.get("media_by_pillar") or {}).items()
            if pillar != "negative"
        ),
        "strict_vlm_queue_items": int(queue_summary.get("strict_vlm_items") or 0),
        "low_level_delta_items": expected,
        "low_level_attempted_items": attempted,
        "low_level_successful_items": successful,
        "low_level_failed_items": attempted - successful,
        "low_level_remaining_items": max(expected - attempted, 0),
        "low_level_progress_fraction": attempted / expected if expected else None,
        "low_level_processes": processes,
        "low_level_complete": complete,
        "low_level_stalled": bool(expected and not complete and processes == 0),
        "policy": "shadow_only_no_accept_reject_or_delete",
    }


def _flat_media_inventory(path: Path) -> dict[str, int | float | None]:
    """Count completed flat media outputs without recursively walking a run."""
    count = 0
    total_bytes = 0
    newest_mtime: float | None = None
    if path.is_dir():
        try:
            entries = path.iterdir()
            for entry in entries:
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                if (
                    entry.is_file()
                    and entry.suffix.lower() in VIDEO_SUFFIXES
                    and stat.st_size > 0
                ):
                    count += 1
                    total_bytes += stat.st_size
                    newest_mtime = max(newest_mtime or stat.st_mtime, stat.st_mtime)
        except OSError:
            pass
    return {
        "files": count,
        "bytes": total_bytes,
        "newest_mtime": newest_mtime,
    }


def witnessed_video_asr_render_status(
    data: Path,
    process_counter=process_count,
    run_name: str = WITNESSED_VIDEO_ASR_RUN,
) -> dict[str, Any] | None:
    """Observe the non-mutating witnessed video+ASR materialization job.

    The expensive directory scans are flat and this function is called only by
    the hourly on-server monitor.  It does not launch, stop, or alter the job.
    """
    run = data / "shadow_scores" / run_name
    if not run.is_dir():
        return None
    spec = read_json(run / "job_spec.json")
    processes = process_counter(
        rf"[m]aterialize_witnessed_reaction_video_asr_windows.py.*{run_name}"
    )
    original_summary = read_json(run / "render_summary.json")
    retry_summary = read_json(run / "render_summary_retry.json")
    coverage_retry_in_progress = bool(
        original_summary and processes and not retry_summary
    )
    summary = (
        {}
        if coverage_retry_in_progress
        else retry_summary or original_summary
    )
    expected_clips = int(
        spec.get("expected_source_clips")
        or summary.get("full_proxies")
        or 0
    )
    expected_candidates = int(
        spec.get("expected_candidate_windows")
        or summary.get("selected_for_shard")
        or 0
    )
    proxies = _flat_media_inventory(run / "full_proxies")
    candidates = _flat_media_inventory(run / "media")
    manifest_path = (
        run / "manifest_retry.jsonl"
        if (run / "manifest_retry.jsonl").is_file()
        else run / "manifest.jsonl"
    )
    failure_path = (
        run / "failures_retry.jsonl"
        if (run / "failures_retry.jsonl").is_file()
        else run / "failures.jsonl"
    )
    manifest_rows = count_lines(manifest_path)
    failure_rows = count_lines(failure_path)
    reported_failures = sum(
        int(summary.get(field) or 0)
        for field in ("failed", "full_proxy_failures", "missing_source_items")
    )
    finalized = bool(summary)
    successfully_complete = bool(
        finalized
        and reported_failures == 0
        and expected_candidates
        and manifest_rows >= expected_candidates
    )
    if successfully_complete:
        phase = "complete"
    elif finalized:
        phase = "complete_with_failures_or_missing"
    elif coverage_retry_in_progress:
        phase = "coverage_retry"
    elif int(candidates["files"]) > 0:
        # Candidate rendering begins only after the attempted proxy pass. Some
        # source proxies may have failed, so proxy count need not reach the
        # input expectation before this phase starts.
        phase = "candidate_window_render"
    elif expected_clips and int(proxies["files"]) < expected_clips:
        phase = "full_clip_proxy_render"
    elif expected_candidates and int(candidates["files"]) < expected_candidates:
        phase = "candidate_window_render"
    else:
        phase = "finalizing_or_unknown"
    return {
        "run": run_name,
        "path": str(run),
        "phase": phase,
        "expected_source_clips": expected_clips,
        "expected_candidate_windows": expected_candidates,
        "full_clip_proxies": proxies,
        "candidate_windows": candidates,
        "manifest_rows": manifest_rows,
        "manifest": manifest_path.name,
        "failure_rows": failure_rows,
        "failures": failure_path.name,
        "proxy_progress_fraction": (
            int(proxies["files"]) / expected_clips if expected_clips else None
        ),
        "candidate_progress_fraction": (
            int(candidates["files"]) / expected_candidates
            if expected_candidates
            else None
        ),
        "processes": processes,
        "complete": finalized,
        "successfully_complete": successfully_complete,
        "reported_failures": reported_failures,
        "coverage_retry_in_progress": coverage_retry_in_progress,
        "stalled": not finalized and processes == 0,
        "summary": summary or None,
        "policy": "shadow_candidate_generation_and_review_ranking_only",
        "automatic_acceptance": False,
    }


def prior_queued_uids(snapshots: Path) -> set[str]:
    queued = set()
    for row in read_jsonl(snapshots):
        for item in row.get("fresh_review_queue") or []:
            if item.get("uid"):
                queued.add(str(item["uid"]))
    return queued


def fresh_review_queue(
    discussion: Path, videos: Path, snapshots: Path, count: int
) -> list[dict[str, Any]]:
    prior = prior_queued_uids(snapshots)
    candidates = sorted(
        (
            path
            for path in videos.iterdir()
            if path.is_file()
            and path.suffix.lower() in VIDEO_SUFFIXES
            and path.stat().st_size > 0
            and path.stem not in prior
        ),
        key=lambda path: (-path.stat().st_mtime_ns, path.name),
    )
    result = []
    for path in candidates[:count]:
        metadata = read_json(discussion / f"{path.stem}.json")
        result.append(
            {
                "uid": path.stem,
                "source": path.stem.split("__", 1)[0],
                "title": metadata.get("title"),
                "media_path": str(path),
                "media_bytes": path.stat().st_size,
                "media_mtime": path.stat().st_mtime,
                "label_status": "source_only_unlabeled",
            }
        )
    return result


def build_snapshot(root: Path, out_dir: Path, sample_size: int) -> dict[str, Any]:
    data = root / "data"
    discussion = data / "discussion"
    videos = data / "discussion_video"
    snapshots = out_dir / "snapshots.jsonl"
    inventory = source_inventory(discussion, videos)
    free_bytes = shutil.disk_usage(videos).free
    backfill_running = process_count("[b]ackfill_commentary_videos.py")
    crawl_running = process_count("[s]rc.search_loop")
    backfill_gaps = backfill_gap_status(
        discussion,
        videos,
        videos / "backfill.jsonl",
    )
    alerts = []
    actionable_gap = sum(
        status["unattempted"] + status["other_latest_status"]
        for status in backfill_gaps.values()
    )
    if actionable_gap and backfill_running == 0:
        alerts.append("dailymotion_rumble_unattempted_backfill_gap_not_running")
    if free_bytes < 10_000_000_000_000:
        alerts.append("discussion_video_filesystem_below_10TB_free")
    general = ledger_summary(videos / "backfill.jsonl")
    youtube = ledger_summary(videos / "youtube_backfill_sk3.jsonl")
    pipeline_status = state_status_counts(data / "state.db")
    audited_search = audited_search_status(data / "state.db")
    shadow_scores = shadow_score_status(data)
    strict_audit_backfill = strict_audit_backfill_status(data)
    witnessed_video_asr = witnessed_video_asr_render_status(data)
    if general["auth_failures_total"] or youtube["auth_failures_total"]:
        alerts.append("authentication_failure_recorded")
    if crawl_running == 0:
        alerts.append("main_crawl_not_running")
    for status, threshold in PIPELINE_BACKLOG_ALERTS.items():
        if pipeline_status.get(status, 0) >= threshold:
            alerts.append(f"{status}_backlog_at_least_{threshold}")
    if shadow_scores:
        for pillar, status in shadow_scores["pillars"].items():
            if status["remaining_items"] and status["processes"] == 0:
                alerts.append(f"{pillar}_shadow_scoring_stalled")
        for pillar, status in (shadow_scores.get("multimodal_v1") or {}).items():
            if status["remaining_items"] and status["processes"] == 0:
                alerts.append(f"{pillar}_multimodal_shadow_scoring_stalled")
    if strict_audit_backfill and strict_audit_backfill["low_level_stalled"]:
        alerts.append("strict_audit_low_level_backfill_stalled")
    if witnessed_video_asr:
        if witnessed_video_asr["stalled"]:
            alerts.append("witnessed_video_asr_render_stalled")
        if witnessed_video_asr["complete"] and not witnessed_video_asr[
            "successfully_complete"
        ]:
            alerts.append("witnessed_video_asr_render_completed_with_failures")
    return {
        "schema_version": 5,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "inventory": inventory,
        "backfill_gaps": backfill_gaps,
        "pipeline_status": pipeline_status,
        "audited_search": audited_search,
        "processes": {
            "commentary_backfill": backfill_running,
            "main_crawl": crawl_running,
        },
        "ledgers": {"general": general, "youtube_sk3": youtube},
        "shadow_scores": shadow_scores,
        "strict_audit_backfill": strict_audit_backfill,
        "witnessed_video_asr_render": witnessed_video_asr,
        "discussion_video_free_bytes": free_bytes,
        "alerts": alerts,
        "fresh_review_queue": fresh_review_queue(
            discussion, videos, snapshots, sample_size
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=4)
    args = parser.parse_args()
    if args.sample_size < 0:
        raise SystemExit("sample-size must be non-negative")
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = (out_dir / "monitor.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("hourly collection monitor is already running") from exc
    snapshot = build_snapshot(root, out_dir, args.sample_size)
    with (out_dir / "snapshots.jsonl").open("a") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n")
    temporary = out_dir / f"latest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, out_dir / "latest.json")
    print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
