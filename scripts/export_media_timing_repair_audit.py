#!/usr/bin/env python3
"""Re-render every timing-flagged corpus clip from raw media for manual audit.

The existing corpus clip is never changed. Repaired proxies are written to a
new audit directory with the source interval and before/after timing metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from src.media_integrity import probe_media_timing, timing_flags
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.media_integrity import probe_media_timing, timing_flags

from scripts.export_commentary_unlabeled_benchmark import render_verified_target


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_raw(raw_dir: Path, uid: str) -> Path:
    matches = [
        path
        for suffix in ("mp4", "mkv", "webm", "mov", "m4v")
        if (path := raw_dir / f"{uid}.{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one raw source for {uid}, found {len(matches)}")
    return matches[0]


def derive_interval(record: dict, corpus_root: Path) -> tuple[str, float, float, dict]:
    relative = Path(record["relative_path"])
    uid = relative.parts[0]
    name = relative.name
    metadata = json.loads((corpus_root / uid / "metadata.json").read_text())
    if record["pillar"] == "instructional":
        match = re.fullmatch(r"demo_(\d+)\.[^.]+", name)
        if not match:
            raise ValueError(f"cannot parse instructional clip index: {name}")
        index = int(match.group(1))
        demo = metadata["demos"][index]
        start = max(0.0, float(demo["start"]) - 1.0)
        end = float(demo["end"]) + 1.0
        label = {
            "clip_index": index,
            "polarity": demo.get("polarity"),
            "norm": demo.get("norm"),
            "explanation": demo.get("explanation"),
        }
    elif record["pillar"] == "witnessed":
        match = re.fullmatch(r"clip_(\d+)\.[^.]+", name)
        if not match:
            raise ValueError(f"cannot parse witnessed clip index: {name}")
        index = int(match.group(1))
        reactions = [
            reaction
            for reaction in metadata.get("reactions") or []
            if int(reaction.get("clip_idx", -1)) == index
        ]
        windows = {
            tuple(float(value) for value in reaction["clip_window"])
            for reaction in reactions
            if reaction.get("clip_window")
        }
        if len(windows) != 1:
            raise ValueError(f"expected one witnessed window for {uid} clip {index}")
        start, end = next(iter(windows))
        label = {
            "clip_index": index,
            "reactions": [
                {
                    "norm": reaction.get("norm"),
                    "tag": reaction.get("tag"),
                    "matched_text": reaction.get("matched_text"),
                }
                for reaction in reactions
            ],
        }
    else:
        raise ValueError(f"unsupported pillar: {record['pillar']}")
    if not 0 <= start < end:
        raise ValueError(f"invalid source interval for {uid}: {start}-{end}")
    return uid, start, end, label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")

    clips_dir = args.out / "repaired_clips"
    clips_dir.mkdir(parents=True)
    output = []
    for ordinal, record in enumerate(
        row for row in load_jsonl(args.timing_jsonl) if row["flagged"]
    ):
        uid, start, end, label = derive_interval(record, args.corpus_root)
        raw = resolve_raw(args.raw_dir, uid)
        repaired = clips_dir / f"{ordinal:02d}__{uid}__{Path(record['relative_path']).stem}.mp4"
        try:
            mode, repaired_timing = render_verified_target(
                raw,
                repaired,
                start,
                end,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                strip_audio=False,
            )
            repaired_flags = timing_flags(repaired_timing)
            repair_error = None
        except RuntimeError as exc:
            mode = None
            repaired_timing = (
                probe_media_timing(repaired, args.ffprobe)
                if repaired.exists()
                else None
            )
            repaired_flags = (
                timing_flags(repaired_timing) if repaired_timing else ["missing_output"]
            )
            repair_error = str(exc)
        output.append(
            {
                "ordinal": ordinal,
                "pillar": record["pillar"],
                "uid": uid,
                "label": label,
                "source_start_sec": start,
                "source_end_sec": end,
                "original_clip": record["path"],
                "original_timing": record["timing"],
                "original_flags": record["flags"],
                "raw_source": str(raw),
                "repaired_clip": str(repaired),
                "repaired_seek_mode": mode,
                "repaired_timing": repaired_timing,
                "repaired_flags": repaired_flags,
                "repair_error": repair_error,
                "corpus_mutated": False,
            }
        )
    manifest = args.out / "manifest.jsonl"
    with manifest.open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"items": len(output), "manifest": str(manifest)}))


if __name__ == "__main__":
    main()
