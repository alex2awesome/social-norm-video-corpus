#!/usr/bin/env python3
"""Harvest yt-dlp .info.json sidecars into a compact metadata ledger.

Every downloaded source leaves an ``{uid}.info.json`` beside the media with
uploader-written title, description, tags, and engagement counts — incident
descriptions included ("Occurred on ... A man attacked ...").  This pass
collects the useful fields for every sidecar into one append-only JSONL so
metadata LFs never need to touch the raw tree.  Nothing is fetched, moved,
or modified.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

HARVEST_VERSION = "harvest_info_sidecars_v1"

FIELDS = (
    "title", "description", "tags", "categories", "uploader", "channel",
    "channel_id", "upload_date", "view_count", "like_count", "comment_count",
    "duration", "webpage_url", "language",
)


def compact_record(uid: str, sidecar: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"uid": uid, "source": uid.split("__")[0]}
    for field in FIELDS:
        value = sidecar.get(field)
        if isinstance(value, str) and len(value) > 4000:
            value = value[:4000]
        if value not in (None, "", []):
            record[field] = value
    record["harvest_version"] = HARVEST_VERSION
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dirs", nargs="+",
                        default=["data/raw_video", "data/discussion_video"])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts: Counter = Counter()
    errors = 0
    with args.out.open("x") as out:
        for directory in args.dirs:
            for path in sorted((args.root / directory).glob("*.info.json")):
                uid = path.name[: -len(".info.json")]
                if uid in seen or "__" not in uid:
                    continue
                seen.add(uid)
                try:
                    sidecar = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                    errors += 1
                    continue
                record = compact_record(uid, sidecar)
                counts[record["source"]] += 1
                counts["with_description"] += "description" in record
                out.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"harvest_version": HARVEST_VERSION,
                      "records": len(seen) - errors, "errors": errors,
                      "counts": dict(counts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
