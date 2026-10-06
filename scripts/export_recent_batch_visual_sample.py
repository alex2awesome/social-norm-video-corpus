#!/usr/bin/env python3
"""Export deterministic three-frame contact sheets from a recent batch run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path


PILLARS = ("instructional", "witnessed", "commentary")


def stable_key(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()


def duration(path: Path, ffprobe: str) -> float:
    out = subprocess.check_output(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)], text=True,
    )
    return float(out.strip())


def numeric_anchors(value) -> list[float]:
    found = []
    if isinstance(value, dict):
        for key in ("start", "start_time", "timestamp"):
            raw = value.get(key)
            if isinstance(raw, (int, float)):
                found.append(float(raw))
        for child in value.values():
            found.extend(numeric_anchors(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(numeric_anchors(child))
    return found


def media_for(repo: Path, pillar: str, uid: str) -> tuple[Path | None, list[float]]:
    if pillar == "instructional":
        files = sorted((repo / "data/instructional" / uid).glob("demo_*.mp4"))
        return (files[0] if files else None), []
    if pillar == "witnessed":
        files = sorted((repo / "data/hits" / uid).glob("clip_*.mp4"))
        return (files[0] if files else None), []
    files = sorted((repo / "data/discussion_video").glob(f"{uid}.*"))
    anchors = []
    meta = repo / "data/discussion" / f"{uid}.json"
    if meta.exists():
        try:
            anchors = numeric_anchors(json.loads(meta.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    return (files[0] if files else None), anchors


def frame_times(dur: float, pillar: str, anchors: list[float]) -> list[float]:
    if pillar == "commentary" and anchors:
        center = min(max(anchors[0], 0.0), max(0.0, dur - 0.1))
        return [max(0.0, center - 4), center, min(max(0.0, dur - 0.1), center + 4)]
    return [dur * 0.25, dur * 0.50, dur * 0.75]


def contact_sheet(media: Path, times: list[float], out: Path, ffmpeg: str) -> None:
    frames = []
    for idx, at in enumerate(times):
        frame = out.with_name(out.stem + f"_{idx}.jpg")
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{at:.3f}",
             "-i", str(media), "-frames:v", "1", "-vf", "scale=320:-2", "-y", str(frame)],
            check=True,
        )
        frames.append(frame)
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error",
         "-i", str(frames[0]), "-i", str(frames[1]), "-i", str(frames[2]),
         "-filter_complex", "hstack=inputs=3", "-frames:v", "1", "-y", str(out)],
        check=True,
    )
    for frame in frames:
        frame.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--processed-since", type=float, required=True)
    ap.add_argument("--per-pillar", type=int, default=4)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--ffprobe", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.repo / "data/state.db")
    conn.row_factory = sqlite3.Row
    manifest = []
    for pillar in PILLARS:
        rows = conn.execute(
            "SELECT video_id,title,query,category,query_source,processed_at "
            "FROM seen_videos WHERE processed_at>=? AND modality=?",
            (args.processed_since, pillar),
        ).fetchall()
        ordered = sorted(rows, key=lambda row: stable_key(row["video_id"]))
        taken = 0
        for row in ordered:
            media, anchors = media_for(args.repo, pillar, row["video_id"])
            if media is None:
                continue
            try:
                dur = duration(media, args.ffprobe)
                times = frame_times(dur, pillar, anchors)
                name = f"{pillar}_{taken:02d}_{row['video_id']}.jpg"
                contact_sheet(media, times, args.out / name, args.ffmpeg)
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
            manifest.append({
                "pillar": pillar, "uid": row["video_id"], "title": row["title"],
                "query": row["query"], "category": row["category"],
                "query_source": row["query_source"], "media": str(media),
                "duration": round(dur, 3), "frame_times": [round(x, 3) for x in times],
                "contact_sheet": name,
            })
            taken += 1
            if taken >= args.per_pillar:
                break
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({"items": len(manifest), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
