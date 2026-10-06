#!/usr/bin/env python3
"""Render low-bandwidth motion/audio proxies for unresolved blind audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_ledger(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def select_unresolved(
    semantic_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    semantic = {row["candidate_id"]: row for row in semantic_rows}
    if len(semantic) != len(semantic_rows):
        raise ValueError("semantic manifest contains duplicate candidate_id")
    ledger_ids = [row["candidate_id"] for row in ledger_rows]
    if len(set(ledger_ids)) != len(ledger_ids):
        raise ValueError("ledger contains duplicate candidate_id")
    missing = set(ledger_ids) - set(semantic)
    if missing:
        raise ValueError(f"ledger IDs absent from semantic manifest: {sorted(missing)}")
    selected = [
        {
            **semantic[row["candidate_id"]],
            "blind_visual_status": row["visual_status"],
            "blind_literal_description": row["literal_description"],
            "blind_failure_or_followup": row["failure_or_followup"],
        }
        for row in ledger_rows
        if row["visual_status"] == "U"
    ]
    return sorted(selected, key=lambda row: int(row["audit_index"]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def proxy_command(ffmpeg: Path, source: Path, target: Path) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        "scale=w='min(360,iw)':h=-2,fps=8",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "31",
        "-maxrate",
        "140k",
        "-bufsize",
        "280k",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "32k",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-movflags",
        "+faststart",
        "-y",
        str(target),
    ]


def render_one(
    row: dict[str, Any],
    out_dir: Path,
    ffmpeg: Path,
    resume: bool,
) -> dict[str, Any]:
    source = Path(row.get("source_path_resolved") or row["source_path"])
    target = out_dir / f"{row['candidate_id']}.mp4"
    if resume and target.is_file() and target.stat().st_size > 0:
        return {
            **row,
            "proxy_path": str(target),
            "proxy_bytes": target.stat().st_size,
            "proxy_sha256": sha256(target),
            "render_status": "reused",
        }
    if not source.is_file():
        return {**row, "render_status": "missing_source"}
    temporary = target.with_suffix(".partial.mp4")
    command = proxy_command(ffmpeg, source, temporary)
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode:
            temporary.unlink(missing_ok=True)
            return {
                **row,
                "render_status": "ffmpeg_error",
                "ffmpeg_returncode": result.returncode,
                "ffmpeg_stderr": result.stderr[-2000:],
            }
        if not temporary.is_file() or temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            return {**row, "render_status": "empty_proxy"}
        os.replace(temporary, target)
        return {
            **row,
            "proxy_path": str(target),
            "proxy_bytes": target.stat().st_size,
            "proxy_sha256": sha256(target),
            "render_status": "rendered",
        }
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        return {
            **row,
            "render_status": "exception",
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--blind-ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    selected = select_unresolved(
        load_jsonl(args.semantic_manifest),
        load_ledger(args.blind_ledger),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    proxies = args.out / "proxies"
    proxies.mkdir(exist_ok=True)
    completed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(render_one, row, proxies, args.ffmpeg, args.resume): row
            for row in selected
        }
        for count, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            completed[int(row["audit_index"])] = future.result()
            if count % 10 == 0 or count == len(futures):
                print(f"{count}/{len(futures)} proxies processed", flush=True)

    rows = [completed[index] for index in sorted(completed)]
    manifest = args.out / "motion_proxy_manifest.jsonl"
    write_jsonl(manifest, rows)
    summary = {
        "kind": "commentary_unresolved_motion_proxies_v1",
        "selected_unresolved": len(selected),
        "successful_proxies": sum(
            row["render_status"] in {"rendered", "reused"} for row in rows
        ),
        "failed_proxies": sum(
            row["render_status"] not in {"rendered", "reused"} for row in rows
        ),
        "total_proxy_bytes": sum(row.get("proxy_bytes", 0) for row in rows),
        "semantic_manifest_sha256": sha256(args.semantic_manifest),
        "blind_ledger_sha256": sha256(args.blind_ledger),
        "motion_proxy_manifest_sha256": sha256(manifest),
        "corpus_mutated": False,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["failed_proxies"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
