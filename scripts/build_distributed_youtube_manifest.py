#!/usr/bin/env python3
"""Build a frozen, deterministic three-host commentary YouTube manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
BUCKETS = 20
WORKER_BUCKETS = {
    "sk1": {0},                 # 5%; disk is nearly full
    "sk2": set(range(1, 11)),  # 50%
    "sk3": set(range(11, 20)), # 45%
}


def uid_bucket(uid: str) -> int:
    digest = hashlib.sha256(uid.encode()).digest()
    return int.from_bytes(digest[:8], "big") % BUCKETS


def assigned_worker(uid: str) -> str:
    bucket = uid_bucket(uid)
    for worker, buckets in WORKER_BUCKETS.items():
        if bucket in buckets:
            return worker
    raise AssertionError(f"unassigned bucket: {bucket}")


def stable_uids(stable_dir: Path | None) -> set[str]:
    if stable_dir is None or not stable_dir.exists():
        return set()
    return {
        path.stem for path in stable_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        and path.stat().st_size > 0
    }


def build_records(
    discussion_dir: Path,
    present: set[str],
) -> list[dict]:
    records = []
    for path in discussion_dir.glob("youtube__*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        uid = str(data.get("video_id") or path.stem)
        url = data.get("url")
        if uid in present or not url:
            continue
        records.append({
            "uid": uid,
            "url": url,
            "source": "youtube",
            "worker": assigned_worker(uid),
            "bucket": uid_bucket(uid),
        })
    return sorted(records, key=lambda row: hashlib.sha256(row["uid"].encode()).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discussion-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stable-dir", type=Path)
    parser.add_argument("--extra-present-uid", action="append", default=[])
    args = parser.parse_args()

    present = stable_uids(args.stable_dir) | set(args.extra_present_uid)
    records = build_records(args.discussion_dir, present)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(json.dumps({
        "records": len(records),
        "already_present": len(present),
        "by_worker": dict(sorted(Counter(r["worker"] for r in records).items())),
        "buckets": {key: sorted(value) for key, value in WORKER_BUCKETS.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
