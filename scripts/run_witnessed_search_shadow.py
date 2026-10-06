#!/usr/bin/env python3
"""Enumerate a frozen pillar-search comparison without mutating crawl state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.search_loop import _junk_title
from src.sources import search_dailymotion
from src.state import load_config


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    shadow = yaml.safe_load(args.shadow.read_text())
    if shadow.get("active") is not False:
        raise SystemExit("refusing to run a search shadow unless active=false")
    if shadow.get("platform") != "dailymotion":
        raise SystemExit("search shadows currently support only Dailymotion")
    version = str(shadow.get("version") or "").strip()
    if re.fullmatch(r"[a-z][a-z0-9_]*_search_shadow_v[0-9]+", version) is None:
        raise SystemExit("missing or invalid pillar search shadow version")

    cfg = load_config()
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("search", {})["results_per_query"] = int(
        shadow.get("results_per_query", 20)
    )
    conn = sqlite3.connect(f"file:{args.state_db}?mode=ro", uri=True)
    seen = {
        row[0]
        for row in conn.execute("SELECT video_id FROM seen_videos")
    }

    records = []
    for group, queries in shadow["groups"].items():
        for query in queries:
            candidates, next_cursor = search_dailymotion(query, cfg, cursor=None)
            for rank, candidate in enumerate(candidates):
                record = {
                    "group": group,
                    "query": query,
                    "rank": rank,
                    "uid": candidate["uid"],
                    "url": candidate["url"],
                    "title": candidate.get("title"),
                    "channel": candidate.get("channel"),
                    "duration": candidate.get("duration"),
                    "already_seen": candidate["uid"] in seen,
                    "title_skipped": bool(_junk_title(candidate)),
                }
                record["candidate_sha256"] = sha256_json(record)
                records.append(record)
            records.append({
                "group": group,
                "query": query,
                "enumeration_summary": True,
                "returned": len(candidates),
                "next_cursor": next_cursor,
            })

    payload = {
        "kind": version,
        "created_at": time.time(),
        "shadow_sha256": hashlib.sha256(args.shadow.read_bytes()).hexdigest(),
        "database_open_mode": "read_only",
        "records": records,
    }
    payload["content_sha256"] = sha256_json(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
