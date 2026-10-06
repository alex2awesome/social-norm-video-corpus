#!/usr/bin/env python3
"""Render hash-verified dense temporal pages for specified audit windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from export_witnessed_search_video_audit import file_sha256, make_sheet  # noqa: E402


def dense_window_key(row: dict, start: float, end: float, spec_index: int) -> str:
    return (
        f"{row['ordinal']:02d}_{row['uid']}_"
        f"{start:g}_{end:g}_{spec_index:02d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    specs = json.loads(args.specs.read_text())
    by_ordinal = {row["ordinal"]: row for row in manifest["records"]}
    frames_root = args.out / "frames"
    sheets_root = args.out / "sheets"
    frames_root.mkdir(parents=True, exist_ok=True)
    sheets_root.mkdir(parents=True, exist_ok=True)
    output = []
    for spec_index, spec in enumerate(specs):
        row = by_ordinal[spec["ordinal"]]
        media = args.artifact_root / row["media_path"]
        if file_sha256(media) != row["media_sha256"]:
            raise SystemExit(f"media hash mismatch: {row['uid']}")
        start, end, count = float(spec["start"]), float(spec["end"]), int(spec["frames"])
        if not (0 <= start < end <= float(row["probed_duration"]) + 0.05):
            raise SystemExit(f"invalid window for {row['uid']}: {start}-{end}")
        window_key = dense_window_key(row, start, end, spec_index)
        item_dir = frames_root / window_key
        item_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(count):
            timestamp = start + (end - start) * (index + 0.5) / count
            target = item_dir / f"f{index:03d}.jpg"
            actual_timestamp = None
            errors = []
            for backoff in (0.0, 0.25, 0.5, 1.0, 2.0):
                candidate_timestamp = max(start, timestamp - backoff)
                target.unlink(missing_ok=True)
                result = subprocess.run([
                    args.ffmpeg, "-hide_banner", "-loglevel", "error", "-ss",
                    f"{candidate_timestamp:.3f}", "-i", str(media), "-frames:v", "1",
                    "-vf", "scale=768:-2", "-strict", "unofficial", "-q:v", "3",
                    "-threads", "1", "-y", str(target),
                ], capture_output=True, text=True)
                if result.returncode == 0 and target.is_file() and target.stat().st_size > 0:
                    actual_timestamp = candidate_timestamp
                    break
                errors.append(result.stderr.strip()[-500:])
            if actual_timestamp is None:
                raise RuntimeError(
                    f"failed to extract {row['uid']} near {timestamp:.3f}s: {errors[-1]}"
                )
            frames.append({
                "frame_index": index,
                "timestamp": round(actual_timestamp, 3),
                "requested_timestamp": round(timestamp, 3),
                "path": target.name,
            })
        pages = []
        for page_index in range(0, len(frames), 16):
            page = sheets_root / f"{window_key}_p{page_index//16:02d}.jpg"
            make_sheet(frames[page_index:page_index + 16], item_dir, page)
            pages.append({"path": str(page.relative_to(args.out)), "sha256": file_sha256(page)})
        output.append({**spec, "uid": row["uid"], "media_sha256": row["media_sha256"],
                       "frame_records": frames, "pages": pages})
    result = {"kind": "dense_video_window_audit", "records": output}
    result["records_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
