#!/usr/bin/env python3
"""Render the complete fresh-review queue from an hourly monitor snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from export_visual_audit_batch import extract_frames, sha256_json
from export_witnessed_search_video_audit import file_sha256, make_sheet


def select_queue(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = snapshot.get("fresh_review_queue") or []
    uids = [str(row.get("uid") or "") for row in rows]
    if any(not uid for uid in uids):
        raise ValueError("fresh review row lacks uid")
    if len(set(uids)) != len(uids):
        raise ValueError("fresh review queue contains duplicate uid")
    for row in rows:
        media = Path(str(row.get("media_path") or ""))
        if not media.is_absolute():
            raise ValueError(f"media_path must be absolute for {row['uid']}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()
    if args.max_frames < 3:
        raise SystemExit("--max-frames must be at least 3")
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")

    snapshot = json.loads(args.snapshot.read_text())
    queue = select_queue(snapshot)
    frames_dir = args.out / "frames"
    sheets_dir = args.out / "sheets"
    frames_dir.mkdir(parents=True)
    sheets_dir.mkdir()

    records = []
    for ordinal, source in enumerate(queue):
        media = Path(source["media_path"])
        if not media.is_file():
            raise FileNotFoundError(media)
        duration, frames = extract_frames(
            args.ffmpeg,
            args.ffprobe,
            media,
            frames_dir,
            ordinal,
            args.max_frames,
        )
        sheet = sheets_dir / f"{ordinal:02d}_{source['uid']}.jpg"
        make_sheet(frames, frames_dir, sheet)
        records.append(
            {
                **source,
                "ordinal": ordinal,
                "duration": round(duration, 3),
                "frames": frames,
                "sheet_path": str(sheet.relative_to(args.out)),
                "sheet_sha256": file_sha256(sheet),
            }
        )

    manifest = {
        "kind": "hourly_monitor_fresh_review_render",
        "checked_at": snapshot.get("checked_at"),
        "snapshot_sha256": file_sha256(args.snapshot),
        "records": records,
        "records_sha256": sha256_json(records),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "frames": sum(len(row["frames"]) for row in records),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
