#!/usr/bin/env python3
"""Render dense temporal evidence for manually screened hourly video sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_rows(manifest: dict, reviews: list[dict]) -> list[tuple[dict, dict]]:
    sources = manifest.get("visual_samples") or []
    by_index = {int(row["audit_index"]): row for row in sources}
    review_by_index = {int(row["audit_index"]): row for row in reviews}
    if len(by_index) != len(sources) or len(review_by_index) != len(reviews):
        raise ValueError("duplicate audit_index")
    if set(by_index) != set(review_by_index):
        raise ValueError("manual review must cover complete source manifest")
    selected = []
    for index in sorted(by_index):
        source, review = by_index[index], review_by_index[index]
        if source["uid"] != review["uid"]:
            raise ValueError(f"uid mismatch at audit_index {index}")
        if review["dense_followup"]:
            selected.append((source, review))
    return selected


def probe_duration(ffprobe: str, media: Path) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(media),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def render_frames(
    ffmpeg: str,
    media: Path,
    duration: float,
    count: int,
    frames_dir: Path,
    prefix: str,
) -> list[dict]:
    records = []
    for index in range(count):
        timestamp = duration * (index + 0.5) / count
        target = frames_dir / f"{prefix}_{index:03d}.jpg"
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-fflags",
                "+discardcorrupt",
                "-i",
                str(media),
                "-ss",
                f"{timestamp:.3f}",
                "-frames:v",
                "1",
                "-vf",
                "scale=640:-2",
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
        if result.returncode or not target.is_file() or not target.stat().st_size:
            raise RuntimeError(
                f"failed to render {media} at {timestamp:.3f}: {result.stderr[-500:]}"
            )
        records.append(
            {
                "frame_index": index,
                "timestamp": round(timestamp, 3),
                "path": f"frames/{target.name}",
            }
        )
    return records


def render_sheet(root: Path, row: dict, target: Path) -> None:
    frames = row["frames"]
    columns, cell_width, cell_height, header = 4, 320, 205, 64
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * cell_width, header + rows * cell_height),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (6, 6),
        f"{row['ordinal']:02d} {row['uid']} | {row.get('title') or ''}",
        fill="white",
    )
    draw.text(
        (6, 26),
        f"norm: {row.get('norm') or ''} | reaction: {row.get('reaction') or ''}",
        fill=(220, 220, 120),
    )
    draw.text((6, 46), row["manual_source_evidence"], fill=(180, 220, 255))
    for index, frame in enumerate(frames):
        image = Image.open(root / frame["path"]).convert("RGB")
        image.thumbnail((cell_width, 180))
        x0 = (index % columns) * cell_width
        y0 = header + (index // columns) * cell_height
        sheet.paste(
            image,
            (x0 + (cell_width - image.width) // 2, y0 + (180 - image.height) // 2),
        )
        draw.text(
            (x0 + 4, y0 + 183),
            f"{frame['frame_index']:02d} t={frame['timestamp']:.2f}s",
            fill="white",
        )
    sheet.save(target, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--max-frames", type=int, default=24)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)
    frames_dir, sheets_dir = args.out / "frames", args.out / "sheets"
    frames_dir.mkdir()
    sheets_dir.mkdir()
    selected = select_rows(
        json.loads(args.manifest.read_text()),
        load_jsonl(args.manual_review),
    )
    records = []
    for ordinal, (source, review) in enumerate(selected):
        media = Path(source["clip_path"])
        if not media.is_absolute():
            media = args.project_root / media
        duration = probe_duration(args.ffprobe, media)
        count = min(args.max_frames, max(8, round(duration * 2)))
        frames = render_frames(
            args.ffmpeg,
            media,
            duration,
            count,
            frames_dir,
            f"{ordinal:02d}_{source['uid']}",
        )
        row = {
            **source,
            "ordinal": ordinal,
            "source_audit_index": source["audit_index"],
            "manual_source_evidence": review["evidence"],
            "duration": round(duration, 3),
            "frames": frames,
        }
        sheet = sheets_dir / f"{ordinal:02d}_{source['uid']}.jpg"
        render_sheet(args.out, row, sheet)
        row["sheet_path"] = str(sheet.relative_to(args.out))
        records.append(row)
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "hourly_dense_video_audit",
                "records": records,
                "records_sha256": digest,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "frames": sum(len(row["frames"]) for row in records),
            }
        )
    )


if __name__ == "__main__":
    main()
