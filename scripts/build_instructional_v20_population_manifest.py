#!/usr/bin/env python3
"""Build the source-disjoint V20 instructional shadow-scoring population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.benchmark_instructional_storyboard_clip import sha256
except ModuleNotFoundError:
    from benchmark_instructional_storyboard_clip import sha256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def build(
    source_rows: list[dict[str, Any]],
    audited_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audited_uids = {str(row["uid"]) for row in audited_rows}
    output = []
    seen_uids: set[str] = set()
    for row in source_rows:
        uid = str(row["uid"])
        if uid in audited_uids:
            continue
        if uid in seen_uids:
            raise ValueError(f"duplicate source UID: {uid}")
        seen_uids.add(uid)
        output.append(
            {
                **row,
                "source_sha256": row.get("source_sha256")
                or row.get("source_clip_sha256"),
                "shadow_population": "v20_source_disjoint_unaudited",
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audited", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1263)
    args = parser.parse_args()

    rows = build(read_jsonl(args.source), read_jsonl(args.audited))
    if len(rows) != args.expected:
        raise ValueError(f"expected {args.expected} rows, got {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary = {
        "kind": "instructional_v20_source_disjoint_shadow_population",
        "status": "unaudited_shadow_scoring_only",
        "policy": "no_keep_reject_or_corpus_mutation_authority",
        "rows": len(rows),
        "distinct_uids": len({row["uid"] for row in rows}),
        "excluded_audited_uids": len(
            {str(row["uid"]) for row in read_jsonl(args.audited)}
        ),
        "artifact_sha256": {
            "source": sha256(args.source),
            "audited": sha256(args.audited),
            "output": sha256(args.output),
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
