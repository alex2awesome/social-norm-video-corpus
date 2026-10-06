#!/usr/bin/env python3
"""Render paginated contact sheets from an exported visual-audit manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_witnessed_search_video_audit import file_sha256, make_sheet


def normalized_frames(source_frames: list[dict]) -> list[dict]:
    frames = []
    for source_frame in source_frames:
        frame = dict(source_frame)
        if "timestamp" not in frame:
            frame["timestamp"] = frame.get(
                "source_timestamp", frame.get("clip_timestamp", 0.0)
            )
        frames.append(frame)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--page-size", type=int, default=16)
    args = parser.parse_args()
    if args.page_size <= 0:
        raise SystemExit("page-size must be positive")

    manifest = json.loads((args.batch / "manifest.json").read_text())
    frames_dir = args.batch / "frames"
    sheets_dir = args.batch / "sheets"
    sheets_dir.mkdir(exist_ok=True)
    records = []
    for item in manifest.get("items") or []:
        ordinal = int(item["ordinal"])
        uid = str(item["uid"])
        # Exporters use either absolute source time, clip-relative time, or the
        # generic timestamp key. Contact sheets only need one explicit time axis.
        frames = normalized_frames(item.get("frames") or [])
        pages = []
        for start in range(0, len(frames), args.page_size):
            target = sheets_dir / f"{ordinal:04d}_{uid}_p{start // args.page_size:02d}.jpg"
            make_sheet(frames[start : start + args.page_size], frames_dir, target)
            pages.append(
                {
                    "path": str(target.relative_to(args.batch)),
                    "sha256": file_sha256(target),
                    "first_frame_index": frames[start]["frame_index"],
                    "last_frame_index": frames[min(start + args.page_size, len(frames)) - 1][
                        "frame_index"
                    ],
                }
            )
        records.append(
            {
                "ordinal": ordinal,
                "item_id": item.get("item_id") or item.get("candidate_id"),
                "uid": uid,
                "frame_manifest_sha256": item["frame_manifest_sha256"],
                "pages": pages,
            }
        )
    output = {
        "kind": "visual_audit_contact_sheets",
        "source_batch_id": manifest.get("batch_id"),
        "page_size": args.page_size,
        "items": records,
    }
    (sheets_dir / "manifest.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"items": len(records), "pages": sum(len(x["pages"]) for x in records)}))


if __name__ == "__main__":
    main()
