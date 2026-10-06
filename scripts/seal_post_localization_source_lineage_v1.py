#!/usr/bin/env python3
"""Hash-seal exact source media before rendering post-localization candidates.

This is a lineage operation only.  It never approves a clip, changes media, or
alters the corpus.  A previously supplied hash must match; an unsealed plan gets
the observed source hash and remains an unapproved render candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = [str(row.get("candidate_id") or "") for row in rows]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("source plan has missing or duplicate candidate ids")
    output = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if row.get("approval_status") != "unreviewed_localization_candidate":
            raise ValueError(f"{candidate_id}: source plan is incorrectly pre-approved")
        source = Path(str(row.get("source_path") or ""))
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"{candidate_id}: missing source media {source}")
        observed = sha256(source)
        expected = row.get("source_sha256")
        if expected and expected != observed:
            raise ValueError(f"{candidate_id}: pre-existing source hash mismatch")
        value = dict(row)
        value.update({
            "source_path": str(source.resolve()),
            "source_sha256": observed,
            "source_lineage_sealed": True,
            "source_lineage_seal_basis": "full_file_sha256_before_final_render",
            "automatic_acceptance": False,
            "corpus_mutation_authorized": False,
        })
        output.append(value)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = seal(read_jsonl(args.plan))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(json.dumps({
        "source_candidates_hash_sealed": len(rows),
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
