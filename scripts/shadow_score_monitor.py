#!/usr/bin/env python3
"""Write an observational progress snapshot for a corpus shadow-score run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # An actively appended final line may be observed mid-write.
            continue
    return rows


def result_progress(path: Path, success_field: str = "result") -> dict[str, int]:
    rows = load_jsonl(path)
    latest = {row.get("item_id"): row for row in rows if row.get("item_id")}
    successful = sum(
        row.get("error") is None and row.get(success_field) is not None
        for row in latest.values()
    )
    return {
        "attempts": len(rows),
        "unique_items": len(latest),
        "successful": successful,
        "latest_failures": len(latest) - successful,
    }


def render_progress(path: Path) -> dict[str, int]:
    return result_progress(path, success_field="proxy_clip")


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or run_dir / "monitor").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = run_dir / "source_manifest.jsonl"
    expected = len(load_jsonl(source_manifest))
    proxy = render_progress(run_dir / "proxy_render_records.jsonl")
    text = result_progress(run_dir / "text_gate_v2.jsonl")
    qwen = result_progress(run_dir / "qwen_v4.jsonl")
    glm = result_progress(run_dir / "glm_v4.jsonl")
    proxy_complete = (run_dir / "proxy_manifest.jsonl").is_file()
    alerts = []
    if proxy["latest_failures"]:
        alerts.append(f"{proxy['latest_failures']} latest proxy render failures")
    if text["latest_failures"]:
        alerts.append(f"{text['latest_failures']} latest text-score failures")
    if qwen["latest_failures"]:
        alerts.append(f"{qwen['latest_failures']} latest Qwen VLM failures")
    if glm["latest_failures"]:
        alerts.append(f"{glm['latest_failures']} latest GLM VLM failures")

    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "schema_version": 1,
        "kind": "shadow_score_progress",
        "checked_at": now.isoformat(),
        "run_dir": str(run_dir),
        "expected_items": expected,
        "proxy_complete_manifest": proxy_complete,
        "progress": {
            "proxy": proxy,
            "text": text,
            "qwen": qwen,
            "glm": glm,
        },
        "complete": (
            expected > 0
            and proxy["successful"] == expected
            and text["successful"] == expected
            and qwen["successful"] == expected
            and glm["successful"] == expected
        ),
        "alerts": alerts,
        "corpus_mutated": False,
    }
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    atomic_write(out_dir / f"{stamp}.json", content)
    atomic_write(out_dir / "latest.json", content)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
