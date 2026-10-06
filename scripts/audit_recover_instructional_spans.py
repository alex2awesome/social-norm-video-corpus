#!/usr/bin/env python3
"""Shadow-only recovery audit for instructional demos with no clip.

The current locator searches the start and end quotes independently from the
beginning of the transcript. Repeated language can therefore produce reversed
spans. This script finds ordered quote pairs and reports recoverable intervals;
it never cuts video or changes metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


RAW_EXTENSIONS = {".mp4", ".mkv", ".webm", ".m4a", ".mov"}


def normalize_token(value: str) -> str:
    return re.sub(r"[^\w']+", "", value.lower(), flags=re.UNICODE)


def quote_tokens(value: str) -> list[str]:
    return [token for part in (value or "").split() if (token := normalize_token(part))]


def word_tokens(transcript: dict[str, Any]) -> list[tuple[str, float, float]]:
    values = transcript.get("words") or []
    if not values:
        values = [word for segment in transcript.get("segments") or [] for word in segment.get("words") or []]
    result = []
    for word in values:
        token = normalize_token(str(word.get("word") or ""))
        start, end = word.get("start"), word.get("end")
        if not token or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        result.append((token, float(start), float(end)))
    return result


def longest_prefix_matches(quote: str, words: list[tuple[str, float, float]]) -> tuple[int, int, list[int]]:
    wanted = quote_tokens(quote)
    haystack = [word[0] for word in words]
    for length in range(len(wanted), 1, -1):
        prefix = wanted[:length]
        matches = [
            index
            for index in range(len(haystack) - length + 1)
            if haystack[index : index + length] == prefix
        ]
        if matches:
            return len(wanted), length, matches
    return len(wanted), 0, []


def recover_span(
    start_quote: str,
    end_quote: str,
    words: list[tuple[str, float, float]],
    max_duration: float,
) -> dict[str, Any]:
    start_total, start_length, starts = longest_prefix_matches(start_quote, words)
    end_total, end_length, ends = longest_prefix_matches(end_quote, words)
    candidates = []
    for start_index in starts:
        for end_index in ends:
            end_word_index = end_index + end_length - 1
            if end_word_index < start_index:
                continue
            start_sec = words[start_index][1]
            end_sec = words[end_word_index][2]
            duration = end_sec - start_sec
            if duration <= 0 or duration > max_duration:
                continue
            candidates.append(
                {
                    "start_word_index": start_index,
                    "end_word_index": end_word_index,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "duration_sec": round(duration, 3),
                }
            )
    candidates.sort(key=lambda row: (row["duration_sec"], row["start_word_index"]))
    best = candidates[0] if candidates else None
    exact = start_length == start_total and end_length == end_total and start_length >= 2 and end_length >= 2
    if best is None:
        confidence = "unresolved"
    elif exact and len(candidates) == 1:
        confidence = "exact_unique"
    elif exact:
        confidence = "exact_ambiguous"
    else:
        confidence = "prefix_ambiguous" if len(candidates) > 1 else "prefix_unique"
    return {
        "confidence": confidence,
        "start_quote_tokens": start_total,
        "start_matched_tokens": start_length,
        "start_occurrences": len(starts),
        "end_quote_tokens": end_total,
        "end_matched_tokens": end_length,
        "end_occurrences": len(ends),
        "ordered_candidates": len(candidates),
        "best": best,
    }


def valid_span(start: Any, end: Any) -> bool:
    return (
        isinstance(start, (int, float))
        and isinstance(end, (int, float))
        and math.isfinite(start)
        and math.isfinite(end)
        and start >= 0
        and end > start
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-duration", type=float, default=180.0)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data = project_root / "data"
    raw_ids = {
        path.stem
        for path in (data / "raw_video").iterdir()
        if path.suffix.lower() in RAW_EXTENSIONS
    }
    counts: Counter[str] = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for metadata_path in (data / "instructional").glob("*/metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError):
                counts["bad_metadata"] += 1
                continue
            uid = str(metadata.get("video_id") or metadata_path.parent.name)
            if uid not in raw_ids:
                continue
            transcript_path = data / "transcripts" / f"{uid}.json"
            try:
                transcript = json.loads(transcript_path.read_text())
            except (OSError, json.JSONDecodeError):
                transcript = {}
            words = word_tokens(transcript)
            for index, demo in enumerate(metadata.get("demos") or []):
                clip_name = demo.get("clip")
                if clip_name and (metadata_path.parent / str(clip_name)).is_file():
                    continue
                if valid_span(demo.get("start"), demo.get("end")):
                    continue
                counts["eligible_invalid_span_raw_present"] += 1
                result = recover_span(
                    str(demo.get("start_quote") or ""),
                    str(demo.get("end_quote") or ""),
                    words,
                    args.max_duration,
                )
                counts[result["confidence"]] += 1
                record = {
                    "item_id": f"instructional:{uid}:{index}",
                    "uid": uid,
                    "item_index": index,
                    "metadata_path": str(metadata_path.relative_to(project_root)),
                    "title": metadata.get("title"),
                    "category": metadata.get("category") or (metadata.get("provenance") or {}).get("category"),
                    "polarity": demo.get("polarity"),
                    "norm": demo.get("norm"),
                    "start_quote": demo.get("start_quote"),
                    "end_quote": demo.get("end_quote"),
                    "original_start": demo.get("start"),
                    "original_end": demo.get("end"),
                    **result,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(dict(sorted(counts.items())), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
