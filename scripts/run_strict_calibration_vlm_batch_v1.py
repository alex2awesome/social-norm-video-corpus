#!/usr/bin/env python3
"""Run the three pillar-specific Qwen calibration rubrics in one GPU lease."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--completion-output", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8271/v1")
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = (
        ("instructional", "v8"),
        ("witnessed", "v7c"),
        ("commentary", "v4"),
    )
    outputs = []
    for pillar, rubric in jobs:
        output = args.out_dir / f"qwen_{pillar}_{rubric}.jsonl"
        outputs.append(output)
        command = [
            args.python,
            str(args.runner),
            "--manifest",
            str(args.manifest),
            "--out",
            str(output),
            "--endpoint",
            args.endpoint,
            "--model",
            args.model,
            "--mode",
            "conditioned",
            "--rubric",
            rubric,
            "--pillar",
            pillar,
            "--workers",
            str(args.workers),
            "--timeout",
            "360",
            "--retries",
            "2",
        ]
        subprocess.run(command, check=True)
    if args.completion_output is not None:
        completed = {}
        for output in outputs:
            for line in output.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("error") is None:
                    completed[row["item_id"]] = {
                        "item_id": row["item_id"],
                        "error": None,
                    }
        expected = {
            json.loads(line)["item_id"]
            for line in args.manifest.read_text().splitlines()
            if line.strip()
        }
        if set(completed) != expected:
            missing = sorted(expected - set(completed))
            raise RuntimeError(f"pillar-specific batch incomplete: {missing[:10]}")
        args.completion_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.completion_output.with_suffix(".tmp")
        temporary.write_text(
            "".join(json.dumps(completed[key], sort_keys=True) + "\n" for key in sorted(completed))
        )
        temporary.replace(args.completion_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
