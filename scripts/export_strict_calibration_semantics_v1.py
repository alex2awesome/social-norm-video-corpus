#!/usr/bin/env python3
"""Export compact post-reveal semantics for a frozen strict calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def excerpt(path: Path, start: float, end: float) -> tuple[str, list[dict[str, Any]]]:
    if not path.is_file():
        return "", []
    payload = json.loads(path.read_text())
    rows = []
    for raw in payload.get("segments") or []:
        try:
            seg_start = float(raw.get("start", 0))
            seg_end = float(raw.get("end", seg_start))
        except (TypeError, ValueError):
            continue
        if seg_end < start or seg_start > end:
            continue
        rows.append({"start": seg_start, "end": seg_end, "text": str(raw.get("text") or "")})
    return " ".join(row["text"].strip() for row in rows if row["text"].strip()), rows


def titles(db: Path, uids: set[str]) -> dict[str, dict[str, Any]]:
    if not db.is_file() or not uids:
        return {}
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in uids)
    rows = connection.execute(
        f"SELECT video_id,title,channel,query,query_source,category FROM seen_videos WHERE video_id IN ({placeholders})",
        tuple(sorted(uids)),
    )
    result = {str(row["video_id"]): dict(row) for row in rows}
    connection.close()
    return result


def export(root: Path, selection: Path) -> list[dict[str, Any]]:
    selected = load_jsonl(selection)
    title_map = titles(root / "data/state.db", {str(row["uid"]) for row in selected})
    output = []
    for row in selected:
        uid = str(row["uid"])
        index = int(str(row["item_id"]).rsplit(":", 1)[1])
        pillar = row["pillar"]
        if pillar == "instructional":
            metadata = json.loads((root / f"data/instructional/{uid}/metadata.json").read_text())
            item = (metadata.get("demos") or [])[index]
            start, end = float(item["start"]), float(item["end"])
            text, segments = excerpt(root / f"data/transcripts/{uid}.json", max(0, start - 3), end + 3)
            semantic = {
                "demo": item,
                "metadata_title": metadata.get("title"),
                "genre": metadata.get("genre"),
            }
        elif pillar == "witnessed":
            metadata = json.loads((root / f"data/hits/{uid}/metadata.json").read_text())
            item = (metadata.get("reactions") or [])[index]
            start, end = map(float, item["clip_window"])
            text, segments = excerpt(root / f"data/transcripts/{uid}.json", max(0, start - 3), end + 3)
            semantic = {
                "reaction": item,
                "all_clip_reactions": [
                    candidate for candidate in metadata.get("reactions") or []
                    if candidate.get("clip_idx") == item.get("clip_idx")
                ],
                "scene_provenance": (metadata.get("provenance") or {}).get("scene"),
            }
        elif pillar == "commentary":
            metadata = json.loads((root / f"data/discussion/{uid}.json").read_text())
            item = (metadata.get("statements") or [])[index]
            start, end = float(item["start"]), float(item["end"])
            text, segments = excerpt(root / f"data/transcripts/{uid}.json", max(0, start - 15), end + 15)
            semantic = {
                "statement": item,
                "metadata_title": metadata.get("title"),
                "scene_provenance": (metadata.get("provenance") or {}).get("scene"),
            }
        else:
            raise ValueError(f"unexpected pillar: {pillar}")
        output.append(
            {
                "audit_index": row["audit_index"],
                "item_id": row["item_id"],
                "uid": uid,
                "pillar": pillar,
                "title_record": title_map.get(uid, {}),
                "assigned": {
                    "norm": row.get("norm"),
                    "polarity": row.get("polarity"),
                    "reaction_tag": row.get("reaction_tag"),
                    "reaction_text": row.get("reaction_text"),
                    "signal": row.get("signal"),
                    "quote": row.get("quote"),
                },
                "semantic": semantic,
                "transcript_excerpt": text,
                "transcript_segments": segments,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    rows = export(args.root.resolve(), args.selection)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps({"items": len(rows), "sha256": digest, "policy": "read_only_compact_export"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
