#!/usr/bin/env python3
"""Summarize on-demand audit VLM leases from their append-only event ledger."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Iterable


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            events.append(event)
    return events


def summarize_events(
    events: Iterable[dict[str, Any]],
    since_unix: float | None = None,
) -> dict[str, dict[str, Any]]:
    by_profile: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "invocations": 0,
            "loads": 0,
            "completed_clients": 0,
            "failed_or_terminated": 0,
            "skipped_without_gpu": 0,
            "records_added": 0,
            "startup_seconds": 0.0,
            "client_seconds": 0.0,
            "ready_residency_seconds": 0.0,
            "total_gpu_lease_seconds": 0.0,
            "latest_event_unix": None,
            "latest_event": None,
        }
    )
    seen_runs: set[tuple[str, str]] = set()
    for event in events:
        timestamp = float(event.get("timestamp_unix", 0))
        if since_unix is not None and timestamp < since_unix:
            continue
        profile = str(event.get("profile", "unknown"))
        row = by_profile[profile]
        run_key = (profile, str(event.get("run_id", "")))
        if event.get("event") == "invocation_started" and run_key not in seen_runs:
            row["invocations"] += 1
            seen_runs.add(run_key)
        if event.get("event") == "server_started":
            row["loads"] += 1
        elif event.get("event") == "server_ready":
            row["startup_seconds"] += float(event.get("startup_seconds") or 0)
        elif event.get("event") == "client_finished":
            row["completed_clients"] += 1
            row["client_seconds"] += float(event.get("client_seconds") or 0)
        elif event.get("event") == "server_unloaded":
            row["records_added"] += int(event.get("output_records_added") or 0)
            row["ready_residency_seconds"] += float(
                event.get("ready_residency_seconds") or 0
            )
            row["total_gpu_lease_seconds"] += float(
                event.get("total_gpu_lease_seconds") or 0
            )
        elif event.get("event") == "invocation_skipped":
            row["skipped_without_gpu"] += 1
        elif event.get("event") in {
            "invocation_failed",
            "invocation_terminated",
        }:
            row["failed_or_terminated"] += 1
        if row["latest_event_unix"] is None or timestamp >= row["latest_event_unix"]:
            row["latest_event_unix"] = timestamp
            row["latest_event"] = event.get("event")
    return dict(sorted(by_profile.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument(
        "--hours",
        type=float,
        default=24,
        help="lookback window; use 0 for the complete ledger (default: 24)",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hours < 0:
        raise ValueError("--hours cannot be negative")
    since = None
    if args.hours:
        since = (datetime.now(timezone.utc) - timedelta(hours=args.hours)).timestamp()
    summary = summarize_events(read_events(args.events), since)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if not summary:
        print("No managed VLM lifecycle events in the requested window.")
        return 0
    for profile, row in summary.items():
        lease_minutes = row["total_gpu_lease_seconds"] / 60
        ready_minutes = row["ready_residency_seconds"] / 60
        client_minutes = row["client_seconds"] / 60
        latest = (
            datetime.fromtimestamp(
                row["latest_event_unix"],
                timezone.utc,
            ).isoformat()
            if row["latest_event_unix"] is not None
            else "n/a"
        )
        print(
            f"{profile}: {row['loads']} loads / {row['invocations']} invocations; "
            f"{row['records_added']} records; GPU lease {lease_minutes:.1f}m "
            f"(ready {ready_minutes:.1f}m, client {client_minutes:.1f}m); "
            f"{row['skipped_without_gpu']} skipped; "
            f"{row['failed_or_terminated']} failed/terminated; "
            f"latest={latest} {row['latest_event']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
