#!/usr/bin/env python3
"""Export sparse-overview and quote-centered frames for commentary video recovery."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}


def normalize(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower(), flags=re.UNICODE).strip()


def best_quote_span(segments: list[dict[str, Any]], quote: str) -> dict[str, Any] | None:
    """Fuzzily locate a quote in up to four consecutive transcript segments."""
    wanted = normalize(quote)
    if not wanted or not segments:
        return None
    best: tuple[tuple[float, int, int], int, int, str] | None = None
    for start_index in range(len(segments)):
        for width in range(1, min(4, len(segments) - start_index) + 1):
            selected = segments[start_index:start_index + width]
            text = " ".join(str(row.get("text") or "") for row in selected)
            candidate = normalize(text)
            if wanted in candidate:
                score = 1.0
            elif len(candidate) >= 20 and candidate in wanted:
                score = 0.98
            else:
                score = difflib.SequenceMatcher(None, wanted, candidate).ratio()
            quality = (score, -abs(len(candidate) - len(wanted)), -width)
            current = (quality, start_index, width, text)
            if best is None or current[0] > best[0]:
                best = current
    assert best is not None
    quality, start_index, width, text = best
    score = quality[0]
    selected = segments[start_index:start_index + width]
    return {
        "score": round(score, 6),
        "start": float(selected[0]["start"]),
        "end": float(selected[-1]["end"]),
        "text": text.strip(),
    }


def probe_duration(ffprobe: str, media: Path) -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(media)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_media(media_dir: Path, ordinal: int, uid: str) -> Path:
    """Resolve either downloaded QA names or stable retained-video names."""
    prefixed = sorted(
        path
        for path in media_dir.glob(f"{ordinal:02d}_{uid}.*")
        if path.suffix.lower() in VIDEO_SUFFIXES
    )
    direct = sorted(
        path
        for path in media_dir.glob(f"{uid}.*")
        if path.suffix.lower() in VIDEO_SUFFIXES
    )
    matches = prefixed or direct
    if len(matches) != 1:
        raise ValueError(f"expected one media file for {uid}, found {len(matches)}")
    return matches[0]


def sample_window(
    ffmpeg: str,
    media: Path,
    start: float,
    end: float,
    count: int,
    frames_dir: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    records = []
    for index in range(count):
        timestamp = start + (end - start) * (index + 0.5) / count
        target = frames_dir / f"{prefix}_{index:02d}.jpg"
        actual = None
        errors = []
        for backoff in (0.0, 0.25, 0.5, 1.0, 2.0):
            candidate = max(start, timestamp - backoff)
            # Fast input seek can land after a damaged/missing H.264 reference
            # frame. Retry with accurate output seek, which decodes forward
            # from the beginning and recovers many otherwise viewable files.
            for accurate_seek in (False, True):
                target.unlink(missing_ok=True)
                seek = (
                    ["-i", str(media), "-ss", f"{candidate:.3f}"]
                    if accurate_seek
                    else ["-ss", f"{candidate:.3f}", "-i", str(media)]
                )
                result = subprocess.run(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-fflags",
                        "+discardcorrupt",
                        *seek,
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=640:-2",
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
                if (
                    result.returncode == 0
                    and target.is_file()
                    and target.stat().st_size
                ):
                    actual = candidate
                    break
                errors.append(result.stderr[-500:])
            if actual is not None:
                break
        if actual is None:
            raise RuntimeError(f"failed to render {media.name} near {timestamp:.3f}: {errors[-1]}")
        records.append({"frame_index": index, "timestamp": round(actual, 3), "requested_timestamp": round(timestamp, 3), "path": f"frames/{target.name}"})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--transcripts-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--overview-frames", type=int, default=16)
    parser.add_argument("--target-frames", type=int, default=24)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir()

    source = json.loads(args.manifest.read_text())
    records = []
    for ordinal, item in enumerate(source["items"]):
        try:
            media = resolve_media(args.media_dir, ordinal, item["uid"])
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        duration = probe_duration(args.ffprobe, media)
        transcript = json.loads((args.transcripts_dir / f"{item['uid']}.json").read_text())
        segments = transcript.get("segments") or []
        behavior = best_quote_span(segments, item["behavior_evidence_quote"])
        stance = best_quote_span(segments, item["stance_evidence_quote"])
        located = [span for span in (behavior, stance) if span]
        target_start = max(0.0, min(span["start"] for span in located) - 5.0)
        target_end = min(duration, max(span["end"] for span in located) + 5.0)
        overview = sample_window(
            args.ffmpeg, media, 0.0, duration, args.overview_frames, frames_dir, f"{ordinal:02d}_overview"
        )
        target = sample_window(
            args.ffmpeg, media, target_start, target_end, args.target_frames, frames_dir, f"{ordinal:02d}_target"
        )
        records.append(
            {
                **item,
                "ordinal": ordinal,
                "media_name": media.name,
                "media_sha256": sha256(media),
                "duration": round(duration, 3),
                "behavior_match": behavior,
                "stance_match": stance,
                "target_window": [round(target_start, 3), round(target_end, 3)],
                "overview_frames": overview,
                "target_frames": target,
            }
        )
    result = {"audit": "commentary_visual_redownload_v1", "records": records}
    result["records_sha256"] = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"records": len(records), "frames": sum(len(r["overview_frames"]) + len(r["target_frames"]) for r in records)}))


if __name__ == "__main__":
    main()
