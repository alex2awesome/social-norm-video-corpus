#!/usr/bin/env python3
"""Freeze and render every routed output from a scene-oriented query cohort.

Instructional and witnessed records are expanded to every saved clip.
Commentary records are expanded to every extracted statement and rendered over
the preceding context window. The blind manifest contains no query, title,
norm, transcript, polarity, or routing label; those live only in the sealed
manifest used after the visual review is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_suffix(path: Path) -> int:
    match = re.search(r"_(\d+)$", path.stem)
    if not match:
        raise ValueError(f"missing numeric suffix: {path}")
    return int(match.group(1))


def demos_by_clip(demos: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Preserve source-list indexing when invalid demos were not clipped."""
    return {
        str(row.get("clip") or f"demo_{index}.mp4"): row
        for index, row in enumerate(demos)
    }


def probe_duration(ffprobe: str, path: Path) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError(f"invalid duration for {path}")
    return duration


def commentary_interval(
    start: float,
    end: float,
    duration: float,
    pre_seconds: float = 12.0,
    post_seconds: float = 2.0,
) -> tuple[float, float]:
    left = max(0.0, start - pre_seconds)
    right = min(duration, max(end + post_seconds, left + 3.0))
    if right <= left:
        raise ValueError("empty commentary interval")
    return left, right


def sample_timestamps(start: float, end: float, count: int) -> list[float]:
    if count < 3:
        raise ValueError("at least three frames are required")
    width = end - start
    return [start + width * (index + 0.5) / count for index in range(count)]


def extract_frame(
    ffmpeg: str,
    source: Path,
    timestamp: float,
    target: Path,
) -> float:
    errors = []
    for backoff in (0.0, 0.5, 1.0, 2.0, 4.0):
        actual = max(0.0, timestamp - backoff)
        target.unlink(missing_ok=True)
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{actual:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                "scale=768:-2",
                "-strict",
                "unofficial",
                "-threads",
                "1",
                "-q:v",
                "3",
                "-y",
                str(target),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and target.is_file() and target.stat().st_size:
            return actual
        errors.append((result.stderr or result.stdout)[-500:])
    raise RuntimeError(f"frame extraction failed near {timestamp:.3f}: {errors[-1]}")


def resolve_discussion_video(root: Path, uid: str) -> Path:
    matches = [
        path
        for path in (root / "data" / "discussion_video").glob(f"{uid}.*")
        if path.suffix not in {".part", ".ytdl"} and ".unsupported-" not in path.name
    ]
    if len(matches) != 1:
        raise ValueError(f"{uid}: expected one discussion video, found {matches}")
    return matches[0]


def expand_source(root: Path, source: dict[str, Any], ffprobe: str) -> list[dict[str, Any]]:
    uid = str(source["video_id"])
    modality = source["modality"]
    common = {
        "uid": uid,
        "source_modality": modality,
        "title": source.get("title"),
        "query": source.get("query"),
        "query_category": source.get("category"),
    }
    if modality == "instructional":
        metadata_path = root / "data" / "instructional" / uid / "metadata.json"
        metadata = read_json(metadata_path)
        demos = demos_by_clip(metadata.get("demos") or [])
        paths = sorted(
            metadata_path.parent.glob("demo_*.mp4"), key=numeric_suffix
        )
        return [
            {
                **common,
                "item_id": f"fresh_scene:{uid}:instructional:{path.stem}",
                "source_path": str(path),
                "clip_name": path.name,
                "metadata_path": str(metadata_path),
                "semantic_record": demos.get(path.name),
                "media_start_sec": 0.0,
                "media_end_sec": None,
            }
            for path in paths
        ]
    if modality == "witnessed":
        metadata_path = root / "data" / "hits" / uid / "metadata.json"
        metadata = read_json(metadata_path)
        by_clip: dict[int, list[dict[str, Any]]] = {}
        for reaction in metadata.get("reactions") or []:
            by_clip.setdefault(int(reaction["clip_idx"]), []).append(reaction)
        paths = sorted(metadata_path.parent.glob("clip_*.mp4"), key=numeric_suffix)
        return [
            {
                **common,
                "item_id": f"fresh_scene:{uid}:witnessed:{path.stem}",
                "source_path": str(path),
                "clip_name": path.name,
                "metadata_path": str(metadata_path),
                "semantic_record": {
                    "reactions": by_clip.get(numeric_suffix(path), []),
                    "provenance_scene": (metadata.get("provenance") or {}).get("scene"),
                },
                "media_start_sec": 0.0,
                "media_end_sec": None,
            }
            for path in paths
        ]
    if modality == "commentary":
        metadata_path = root / "data" / "discussion" / f"{uid}.json"
        metadata = read_json(metadata_path)
        source_path = resolve_discussion_video(root, uid)
        duration = probe_duration(ffprobe, source_path)
        rows = []
        for index, statement in enumerate(metadata.get("statements") or []):
            start = float(statement["start"])
            end = float(statement["end"])
            left, right = commentary_interval(start, end, duration)
            rows.append(
                {
                    **common,
                    "item_id": f"fresh_scene:{uid}:commentary:statement_{index}",
                    "source_path": str(source_path),
                    "clip_name": source_path.name,
                    "metadata_path": str(metadata_path),
                    "semantic_record": statement,
                    "media_start_sec": left,
                    "media_end_sec": right,
                    "source_duration": duration,
                }
            )
        return rows
    raise ValueError(f"unsupported modality: {modality}")


def load_sources(db_path: Path, query_source: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT video_id,title,query,category,modality,status,processed_at
                FROM seen_videos
                WHERE query_source=? AND status='done' AND modality IS NOT NULL
                ORDER BY modality,video_id
                """,
                (query_source,),
            )
        ]
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--query-source", default="manual_audit_search_v1")
    parser.add_argument("--seed", default="fresh-scene-query-all-v1")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite audit directory: {args.out}")
    if args.frames != 12:
        raise SystemExit("this frozen audit requires exactly 12 frames per item")
    root = args.root.resolve()
    sources = load_sources(root / "data" / "state.db", args.query_source)
    items = [
        item
        for source in sources
        for item in expand_source(root, source, args.ffprobe)
    ]
    random.Random(args.seed).shuffle(items)
    args.out.mkdir(parents=True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir()
    blind_rows = []
    sealed_rows = []
    errors = []
    for audit_index, item in enumerate(items):
        source_path = Path(item["source_path"])
        duration = probe_duration(args.ffprobe, source_path)
        start = float(item.get("media_start_sec") or 0.0)
        end = (
            float(item["media_end_sec"])
            if item.get("media_end_sec") is not None
            else duration
        )
        end = min(duration, end)
        frames = []
        try:
            for frame_index, timestamp in enumerate(
                sample_timestamps(start, end, args.frames)
            ):
                name = f"{audit_index:04d}_f{frame_index:02d}.jpg"
                actual = extract_frame(
                    args.ffmpeg, source_path, min(duration - 0.02, timestamp), frames_dir / name
                )
                frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": round(actual, 3),
                        "path": name,
                        "sha256": file_sha256(frames_dir / name),
                    }
                )
        except Exception as exc:
            errors.append({"audit_index": audit_index, "item_id": item["item_id"], "error": str(exc)})
        blind_rows.append(
            {
                "audit_index": audit_index,
                "item_id": item["item_id"],
                "uid": item["uid"],
                "frames": [row["path"] for row in frames],
                "frame_manifest_sha256": sha256_json(frames),
            }
        )
        sealed_rows.append(
            {
                **item,
                "audit_index": audit_index,
                "probed_duration": duration,
                "sample_interval": [start, end],
                "frames": frames,
            }
        )
        if (audit_index + 1) % 10 == 0 or audit_index + 1 == len(items):
            print(f"{audit_index + 1}/{len(items)} rendered", flush=True)
    write_jsonl(args.out / "blind_manifest.jsonl", blind_rows)
    write_jsonl(args.out / "sealed_selection.jsonl", sealed_rows)
    manifest = {
        "schema_version": 1,
        "kind": "fresh_scene_oriented_query_all_outputs_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_source": args.query_source,
        "seed": args.seed,
        "source_count": len(sources),
        "item_count": len(items),
        "by_modality": {
            modality: sum(item["source_modality"] == modality for item in items)
            for modality in ("instructional", "witnessed", "commentary")
        },
        "blind_fields_excluded": [
            "source_modality",
            "title",
            "query",
            "query_category",
            "semantic_record",
        ],
        "source_rows_sha256": sha256_json(sources),
        "sealed_rows_sha256": sha256_json(sealed_rows),
        "errors": errors,
        "items": blind_rows,
        "policy": "manual_blind_review_before_query_or_keep_rule_changes",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "sources": len(sources),
                "items": len(items),
                "by_modality": manifest["by_modality"],
                "errors": len(errors),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
