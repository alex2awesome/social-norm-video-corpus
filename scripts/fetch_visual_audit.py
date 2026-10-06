#!/usr/bin/env python3
"""Run the frame-only audit sampler on sk3 and copy the small bundle locally."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


REMOTE_ROOT = "/lfs/skampere3/0/alexspan/norm-scraper"
REMOTE_PYTHON = "/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/python"
REMOTE_BIN = "/lfs/skampere3/0/alexspan/envs/norm-scraper/bin"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="sk3")
    parser.add_argument("--since-hours", type=float, default=1.5)
    parser.add_argument("--instructional", type=int, default=24)
    parser.add_argument("--witnessed", type=int, default=16)
    parser.add_argument("--commentary", type=int, default=16)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--local-root", type=Path, default=Path("audit_runs"))
    args = parser.parse_args()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    seed = args.seed or f"hourly-{stamp}"
    remote_out = f"/tmp/norm_visual_audit_{stamp}"
    local_out = args.local_root / f"{stamp}_hourly"
    sampler = Path(__file__).with_name("visual_audit_sample.py").resolve()

    run(
        [
            "scp",
            "-q",
            str(sampler),
            f"{args.host}:{REMOTE_ROOT}/scripts/visual_audit_sample.py",
        ]
    )
    remote_args = [
            REMOTE_PYTHON,
            "scripts/visual_audit_sample.py",
            "--out",
            remote_out,
            "--since-hours",
            str(args.since_hours),
            "--instructional",
            str(args.instructional),
            "--witnessed",
            str(args.witnessed),
            "--commentary",
            str(args.commentary),
            "--seed",
            seed,
            "--ffmpeg",
            f"{REMOTE_BIN}/ffmpeg",
            "--ffprobe",
            f"{REMOTE_BIN}/ffprobe",
    ]
    remote_command = f"cd {shlex.quote(REMOTE_ROOT)} && {shlex.join(remote_args)}"
    run(["ssh", "-o", "BatchMode=yes", args.host, remote_command])
    local_out.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.json", "visual_review.tsv", "commentary_review.tsv", "frames"):
        run(["scp", "-q", "-r", f"{args.host}:{remote_out}/{name}", str(local_out)])
    print(local_out)


if __name__ == "__main__":
    main()
