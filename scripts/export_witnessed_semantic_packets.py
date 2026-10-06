#!/usr/bin/env python3
"""Export post-reveal evidence packets for a frozen witnessed audit selection.

The exporter is read-only with respect to the corpus. It resolves each selected
clip to its detector reaction, clip window, title, and transcript excerpt so a
human can judge the strict witnessed-pillar contract after the visual review
has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_item_id(item_id: str) -> tuple[str, int]:
    parts = item_id.split(":")
    if len(parts) != 3 or parts[0] != "witnessed":
        raise ValueError(f"invalid witnessed item_id: {item_id!r}")
    try:
        reaction_ordinal = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"invalid clip index in item_id: {item_id!r}") from exc
    if reaction_ordinal < 0:
        raise ValueError(f"negative reaction ordinal in item_id: {item_id!r}")
    return parts[1], reaction_ordinal


def find_metadata(root: Path, uid: str) -> Path:
    candidates = [
        root / "data" / "hits" / uid / "metadata.json",
        root / "data" / "hits_instr_quarantine" / uid / "metadata.json",
        root / "data" / "quarantine" / uid / "metadata.json",
    ]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError(f"{uid}: expected exactly one metadata file, found {found}")
    return found[0]


def transcript_excerpt(
    transcript_path: Path, start: float, end: float
) -> tuple[str, list[dict[str, Any]]]:
    if not transcript_path.is_file():
        return "", []
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments: list[dict[str, Any]] = []
    for segment in transcript.get("segments", []):
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))
        if seg_end < start or seg_start > end:
            continue
        text = str(segment.get("text", "")).strip()
        segments.append({"start": seg_start, "end": seg_end, "text": text})
    return " ".join(row["text"] for row in segments if row["text"]), segments


def title_map(db_path: Path, uids: set[str]) -> dict[str, dict[str, Any]]:
    if not db_path.is_file():
        return {}
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in uids)
    if not placeholders:
        return {}
    rows = connection.execute(
        "SELECT video_id, title, channel, url, duration, source, status "
        f"FROM seen_videos WHERE video_id IN ({placeholders})",
        tuple(sorted(uids)),
    ).fetchall()
    connection.close()
    return {
        row["video_id"]: {
            "title": row["title"],
            "channel": row["channel"],
            "url": row["url"],
            "duration": row["duration"],
            "source": row["source"],
            "status": row["status"],
        }
        for row in rows
    }


def export_packets(root: Path, selection_path: Path) -> list[dict[str, Any]]:
    selection = read_jsonl(selection_path)
    ids = [row.get("item_id") for row in selection]
    if len(ids) != len(set(ids)):
        raise ValueError("selection contains duplicate item_id values")

    parsed = [parse_item_id(str(item_id)) for item_id in ids]
    titles = title_map(root / "data" / "state.db", {uid for uid, _ in parsed})
    packets: list[dict[str, Any]] = []
    for selected, (uid, reaction_ordinal) in zip(selection, parsed, strict=True):
        metadata_path = find_metadata(root, uid)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reactions = metadata.get("reactions", [])
        if not isinstance(reactions, list) or reaction_ordinal >= len(reactions):
            raise ValueError(
                f"{selected['item_id']}: reaction ordinal is out of range"
            )
        reaction = reactions[reaction_ordinal]
        clip_idx = reaction.get("clip_idx")
        if not isinstance(clip_idx, int) or clip_idx < 0:
            raise ValueError(f"{selected['item_id']}: invalid clip_idx")
        clip_window = reaction.get("clip_window")
        if (
            not isinstance(clip_window, list)
            or len(clip_window) != 2
            or not all(isinstance(value, (int, float)) for value in clip_window)
        ):
            raise ValueError(f"{selected['item_id']}: invalid clip_window")
        window_start = float(clip_window[0])
        window_end = float(clip_window[1])
        if window_start >= window_end:
            raise ValueError(f"{selected['item_id']}: invalid clip bounds")
        clip_reactions = [
            candidate
            for candidate in reactions
            if candidate.get("clip_idx") == clip_idx
            and candidate.get("clip_window") == clip_window
        ]
        excerpt_text, excerpt_segments = transcript_excerpt(
            root / "data" / "transcripts" / f"{uid}.json",
            max(0.0, window_start - 3.0),
            window_end + 3.0,
        )
        provenance = metadata.get("provenance", {})
        packets.append(
            {
                "item_id": selected["item_id"],
                "audit_index": selected.get("audit_index"),
                "uid": uid,
                "clip_idx": clip_idx,
                "reaction_ordinal": reaction_ordinal,
                "title_record": titles.get(uid, {}),
                "selection": {
                    "assigned_norm": selected.get("norm"),
                    "found_by_query": selected.get("found_by_query"),
                    "query_source": selected.get("query_source"),
                    "score_band": selected.get("score_band"),
                    "activity_percentile": selected.get("activity_percentile"),
                },
                "provenance": {
                    "category": provenance.get("category"),
                    "modality": provenance.get("modality"),
                    "agent": provenance.get("agent"),
                    "found_by_query": provenance.get("found_by_query"),
                    "query_source": provenance.get("query_source"),
                },
                "clip_window": [window_start, window_end],
                "reaction": {
                    "phrase": reaction.get("phrase"),
                    "matched_text": reaction.get("matched_text"),
                    "tier": reaction.get("tier"),
                    "tag": reaction.get("tag"),
                    "start": reaction.get("start"),
                    "end": reaction.get("end"),
                    "context": reaction.get("context"),
                    "speaker": reaction.get("speaker"),
                },
                "clip_reactions": [
                    {
                        "reaction_ordinal": ordinal,
                        "phrase": candidate.get("phrase"),
                        "matched_text": candidate.get("matched_text"),
                        "tier": candidate.get("tier"),
                        "tag": candidate.get("tag"),
                        "start": candidate.get("start"),
                        "end": candidate.get("end"),
                        "context": candidate.get("context"),
                        "speaker": candidate.get("speaker"),
                    }
                    for ordinal, candidate in enumerate(reactions)
                    if candidate in clip_reactions
                ],
                "transcript_excerpt": excerpt_text,
                "transcript_segments": excerpt_segments,
                "metadata_path": str(metadata_path.relative_to(root)),
            }
        )
    return packets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    packets = export_packets(args.root, args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "items": len(packets),
                "selection_sha256": sha256(args.selection),
                "output": str(args.output),
                "policy": "read_only_export",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
