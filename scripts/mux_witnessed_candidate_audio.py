#!/usr/bin/env python3
"""Mux original source audio onto witnessed silent candidate proxies.

The video stream is copied bit-for-bit from the already-rendered candidate
proxy. Audio is cut from the corresponding source window and encoded as AAC.
Outputs are separate shadow artifacts; silent proxies and corpus files are not
modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def attach_source_context(
    rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join raw source paths without changing the candidate representation."""
    context = {str(row["candidate_id"]): row for row in context_rows}
    if len(context) != len(context_rows):
        raise ValueError("source context manifest has duplicate candidate_id")
    missing = [str(row["candidate_id"]) for row in rows if str(row["candidate_id"]) not in context]
    if missing:
        raise ValueError(f"source context missing {len(missing)} candidates")
    return [
        {
            **row,
            "source_clip": context[str(row["candidate_id"])]["source_clip"],
        }
        for row in rows
    ]


def candidate_ordinal(row: dict[str, Any]) -> int:
    for field in ("ordinal", "storyboard_index", "audit_index"):
        if row.get(field) is not None:
            return int(row[field])
    raise ValueError(f"{row.get('candidate_id')}: missing stable ordinal")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_media(row: dict[str, Any], manifest_dir: Path) -> Path:
    supplied = Path(str(row["candidate_video_path"]))
    candidates = [supplied]
    if not supplied.is_absolute():
        candidates.extend((Path.cwd() / supplied, manifest_dir / supplied))
    return next((path.resolve() for path in candidates if path.is_file()), candidates[-1])


def source_start_time(source: Path, ffprobe: str) -> float:
    payload = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries", "format=start_time",
        "-of", "json", str(source),
    ], text=True))
    return max(0.0, float((payload.get("format") or {}).get("start_time") or 0))


def has_audio_stream(source: Path, ffprobe: str) -> bool:
    output = subprocess.check_output([
        ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(source),
    ], text=True)
    return bool(output.strip())


def mux_argv(
    ffmpeg: str,
    silent_video: Path,
    source: Path,
    target: Path,
    *,
    audio_start: float,
    duration: float,
) -> list[str]:
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(silent_video), "-ss", f"{audio_start:.3f}", "-i", str(source),
        "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-shortest",
        "-movflags", "+faststart", str(target),
    ]


def video_only_argv(ffmpeg: str, silent_video: Path, target: Path) -> list[str]:
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(silent_video), "-map", "0:v:0", "-c:v", "copy",
        "-movflags", "+faststart", str(target),
    ]


def mux_one(
    row: dict[str, Any],
    manifest_dir: Path,
    out_dir: Path,
    portable_root: Path,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, Any]:
    silent = resolve_media(row, manifest_dir)
    source = Path(str(row["source_clip"]))
    digest = hashlib.sha256(str(row["candidate_id"]).encode()).hexdigest()[:16]
    try:
        target = out_dir / f"{candidate_ordinal(row):06d}_{digest}.mp4"
        if not silent.is_file():
            raise FileNotFoundError(silent)
        if sha256(silent) != row["candidate_video_sha256"]:
            raise ValueError("silent candidate video hash mismatch")
        if not source.is_file():
            raise FileNotFoundError(source)
        audio_present = has_audio_stream(source, ffprobe)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size == 0:
            if audio_present:
                audio_start = source_start_time(source, ffprobe) + float(
                    row["media_start_sec"]
                )
                argv = mux_argv(
                    ffmpeg,
                    silent,
                    source,
                    target,
                    audio_start=audio_start,
                    duration=float(row["window_duration_sec"]),
                )
            else:
                argv = video_only_argv(ffmpeg, silent, target)
            subprocess.run(argv, check=True)
        try:
            relative = target.resolve().relative_to(portable_root.resolve())
        except ValueError:
            relative = target.resolve()
        return {
            **row,
            "silent_candidate_video_path": row["candidate_video_path"],
            "silent_candidate_video_sha256": row["candidate_video_sha256"],
            "candidate_video_path": str(relative),
            "candidate_video_sha256": sha256(target),
            "audio_present": audio_present,
            "audio_source": "original_source_window" if audio_present else "none",
            "video_stream_policy": "copied_from_silent_candidate_proxy",
            "error": None,
            "policy": "shadow_media_only_preserve_all_inputs",
        }
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {
            **row,
            "candidate_video_path": None,
            "candidate_video_sha256": None,
            "audio_present": None,
            "error": f"{type(exc).__name__}: {exc}",
            "policy": "shadow_media_only_preserve_all_inputs",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--source-context-manifest",
        type=Path,
        help="candidate-keyed manifest supplying raw source_clip paths",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for path in (args.out_manifest, args.failures, args.summary):
        if path.exists():
            raise SystemExit(f"refusing to overwrite: {path}")
    rows = read_jsonl(args.manifest)
    if args.source_context_manifest:
        rows = attach_source_context(rows, read_jsonl(args.source_context_manifest))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                mux_one,
                row,
                args.manifest.parent,
                args.out_dir,
                args.out_manifest.parent,
                args.ffmpeg,
                args.ffprobe,
            )
            for row in rows
        ]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=candidate_ordinal)
    successes = [row for row in results if not row.get("error")]
    failures = [row for row in results if row.get("error")]
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in successes
    ))
    args.failures.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in failures
    ))
    summary = {
        "kind": "witnessed_candidate_original_audio_mux_v1",
        "input_rows": len(rows),
        "rendered": len(successes),
        "failed": len(failures),
        "audio_present": sum(row["audio_present"] is True for row in successes),
        "video_only": sum(row["audio_present"] is False for row in successes),
        "corpus_mutated": False,
        "policy": "shadow_only_preserve_all_input_media_and_metadata",
        "manifest_sha256": sha256(args.out_manifest),
        "failures_sha256": sha256(args.failures),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
