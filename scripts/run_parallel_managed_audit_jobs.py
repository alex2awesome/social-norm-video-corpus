#!/usr/bin/env python3
"""Run independent managed audit leases concurrently without a shell."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def load_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("config requires a non-empty jobs list")
    names: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("each job must be an object")
        name = job.get("name")
        argv = job.get("argv")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("job names must be unique non-empty strings")
        if not isinstance(argv, list) or not argv or any(
            not isinstance(value, str) or not value for value in argv
        ):
            raise ValueError(f"{name}: argv must be a non-empty string list")
        names.add(name)
    return jobs


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    print(f"parallel-audit: starting {job['name']}", flush=True)
    completed = subprocess.run(job["argv"], check=False)
    result = {
        "name": job["name"],
        "argv": job["argv"],
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
    }
    print(
        f"parallel-audit: finished {job['name']} rc={completed.returncode}",
        flush=True,
    )
    return result


def run_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: row["name"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    results = run_jobs(load_jobs(args.config))
    summary = {
        "kind": "parallel_managed_audit_jobs",
        "jobs": results,
        "all_succeeded": all(row["returncode"] == 0 for row in results),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if summary["all_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
