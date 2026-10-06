#!/usr/bin/env python3
"""Verify a ledger-driven, non-overwriting YouTube consolidation.

The distributed workers already recorded the byte size and SHA-256 of every
successful download.  This verifier treats those ledgers as the source of truth
and admits a destination file only when its bytes match.  It never downloads,
moves, deletes, or rewrites media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def downloaded_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "downloaded":
            rows.append(row)
    return rows


def verify(video_dir: Path, ledgers: list[Path]) -> tuple[list[dict], dict]:
    expected: dict[str, dict] = {}
    workers: dict[str, set[str]] = defaultdict(set)
    conflicts: list[dict] = []
    ledger_counts: Counter[str] = Counter()

    for ledger in ledgers:
        for row in downloaded_rows(ledger):
            uid = str(row["uid"])
            worker = str(row.get("worker") or ledger.stem)
            ledger_counts[worker] += 1
            workers[uid].add(worker)
            signature = (int(row["bytes"]), str(row["sha256"]))
            if uid in expected:
                prior = expected[uid]
                prior_signature = (int(prior["bytes"]), str(prior["sha256"]))
                if signature != prior_signature:
                    conflicts.append(
                        {
                            "uid": uid,
                            "prior": prior_signature,
                            "new": signature,
                            "worker": worker,
                        }
                    )
            else:
                expected[uid] = row

    verified: list[dict] = []
    missing: list[str] = []
    size_mismatch: list[dict] = []
    hash_mismatch: list[dict] = []
    for uid, row in sorted(expected.items()):
        filename = Path(str(row.get("path") or f"{uid}.mp4")).name
        target = video_dir / filename
        if not target.is_file():
            missing.append(uid)
            continue
        actual_bytes = target.stat().st_size
        if actual_bytes != int(row["bytes"]):
            size_mismatch.append(
                {
                    "uid": uid,
                    "expected": int(row["bytes"]),
                    "actual": actual_bytes,
                }
            )
            continue
        actual_sha256 = sha256(target)
        if actual_sha256 != str(row["sha256"]):
            hash_mismatch.append(
                {
                    "uid": uid,
                    "expected": str(row["sha256"]),
                    "actual": actual_sha256,
                }
            )
            continue
        verified.append(
            {
                "uid": uid,
                "path": str(target),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "workers": sorted(workers[uid]),
                "source": "youtube",
                "status": "verified_consolidated",
            }
        )

    summary = {
        "kind": "distributed_youtube_consolidation_verification",
        "policy": "non_overwriting_ledger_sha256",
        "ledgers": [str(path) for path in ledgers],
        "ledger_download_rows_by_worker": dict(sorted(ledger_counts.items())),
        "ledger_download_rows": sum(ledger_counts.values()),
        "unique_expected_uids": len(expected),
        "duplicate_uid_rows": sum(len(value) - 1 for value in workers.values()),
        "verified_uids": len(verified),
        "verified_bytes": sum(row["bytes"] for row in verified),
        "missing_uids": missing,
        "size_mismatches": size_mismatch,
        "hash_mismatches": hash_mismatch,
        "ledger_conflicts": conflicts,
        "complete": not (
            missing or size_mismatch or hash_mismatch or conflicts
        ),
        "source_media_mutated": False,
    }
    return verified, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, action="append", required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()
    verified, summary = verify(args.video_dir, args.ledger)
    args.out_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in verified)
    )
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
