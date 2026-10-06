#!/usr/bin/env python3
"""Run several audit clients during one managed VLM lease.

The outer managed-vLLM wrapper owns model startup and shutdown.  This helper
amortizes that startup across independent, resumable audit clients without a
shell or implicit corpus mutations.  Commands are explicit argv arrays in a
JSON file and may use only the ``{endpoint}`` and ``{model}`` placeholders.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


PLACEHOLDERS = {"{endpoint}", "{model}"}


def load_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("bundle requires a non-empty jobs list")
    names: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("each job must be an object")
        name = job.get("name")
        argv = job.get("argv")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("job names must be unique non-empty strings")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) or not value for value in argv)
        ):
            raise ValueError(f"{name}: argv must be a non-empty string list")
        names.add(name)
    return jobs


def render_argv(argv: list[str], endpoint: str, model: str) -> list[str]:
    replacements = {"{endpoint}": endpoint, "{model}": model}
    rendered = []
    for value in argv:
        unknown = [
            token for token in re.findall(r"\{[^{}]+\}", value)
            if token not in PLACEHOLDERS
        ]
        if unknown:
            raise ValueError(f"unsupported placeholder(s): {unknown}")
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        rendered.append(value)
    return rendered


def run_bundle(
    jobs: list[dict[str, Any]], endpoint: str, model: str
) -> list[dict[str, Any]]:
    results = []
    for job in jobs:
        argv = render_argv(job["argv"], endpoint, model)
        started = time.monotonic()
        print(f"audit-bundle: starting {job['name']}", flush=True)
        completed = subprocess.run(argv, check=False)
        results.append(
            {
                "name": job["name"],
                "argv": argv,
                "returncode": completed.returncode,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        print(
            f"audit-bundle: finished {job['name']} rc={completed.returncode}",
            flush=True,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    jobs = load_jobs(args.bundle)
    results = run_bundle(jobs, args.endpoint, args.model)
    summary = {
        "kind": "managed_vlm_audit_bundle",
        "model": args.model,
        "jobs": results,
        "all_succeeded": all(row["returncode"] == 0 for row in results),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if summary["all_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
