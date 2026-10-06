#!/usr/bin/env python3
"""Idempotently seed only manually supported scene-oriented search queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.validate_audited_search_queries_v1 import evaluate, read_jsonl
except ModuleNotFoundError:
    from validate_audited_search_queries_v1 import evaluate, read_jsonl  # type: ignore[no-redef]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_audit(contract: dict[str, Any], contract_path: Path) -> None:
    audit = contract.get("audit") or {}
    if audit.get("validation_required_before_seed") is not True:
        return
    root = contract_path.resolve().parent.parent
    manual_path = root / str(audit.get("source") or "")
    selection_path = root / str(audit.get("selection") or "")
    expected = {
        manual_path: str(audit.get("source_sha256") or ""),
        selection_path: str(audit.get("selection_sha256") or ""),
    }
    for artifact, digest in expected.items():
        if not artifact.is_file():
            raise ValueError(f"missing frozen search audit artifact: {artifact}")
        if len(digest) != 64 or sha256(artifact) != digest:
            raise ValueError(f"frozen search audit hash mismatch: {artifact}")
    evaluate(
        contract,
        json.loads(selection_path.read_text()),
        read_jsonl(manual_path),
    )


def load_contract(path: Path) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text())
    if not isinstance(contract, dict) or not contract.get("active"):
        raise ValueError("audited search contract is absent or inactive")
    queries = contract.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("audited search contract has no queries")
    seen = set()
    for row in queries:
        query = str(row.get("query") or "").strip()
        pillar = row.get("pillar")
        category = str(row.get("category") or "")
        reviewed = int(row.get("reviewed") or 0)
        strict = int(row.get("strict_target_passes") or 0)
        if not query or query in seen:
            raise ValueError(f"blank or duplicate query: {query!r}")
        if pillar not in {"instructional", "commentary"}:
            raise ValueError(f"unsupported pillar: {pillar!r}")
        expected_prefix = "instr_" if pillar == "instructional" else "comm_"
        if not category.startswith(expected_prefix):
            raise ValueError(
                f"{query!r}: category must start with {expected_prefix!r}"
            )
        if reviewed < 1 or strict < 1 or strict > reviewed:
            raise ValueError(f"{query!r}: invalid manual audit counts")
        seen.add(query)
    validate_frozen_audit(contract, path)
    return contract


def existing_queries(
    connection: sqlite3.Connection,
    platform: str,
    queries: list[str],
) -> set[str]:
    return {
        query
        for query in queries
        if connection.execute(
            "SELECT 1 FROM queries WHERE platform=? AND query=?",
            (platform, query),
        ).fetchone()
    }


def plan(
    connection: sqlite3.Connection,
    contract: dict[str, Any],
) -> dict[str, Any]:
    platform = str(contract["platform"])
    queries = [str(row["query"]) for row in contract["queries"]]
    existing = existing_queries(connection, platform, queries)
    return {
        "version": contract["version"],
        "platform": platform,
        "source": contract["source"],
        "priority": float(contract["priority"]),
        "contract_queries": len(queries),
        "already_present": sorted(existing),
        "to_insert": [query for query in queries if query not in existing],
        "policy": "retrieval_only_preserve_existing",
    }


def apply(
    connection: sqlite3.Connection,
    contract: dict[str, Any],
) -> dict[str, Any]:
    before = plan(connection, contract)
    now = time.time()
    inserted = 0
    with connection:
        for row in contract["queries"]:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO queries
                   (platform, query, source, category, priority, active, added_at)
                   VALUES (?, ?, ?, ?, ?, 1, ?)""",
                (
                    contract["platform"],
                    str(row["query"]).strip(),
                    contract["source"],
                    row["category"],
                    float(contract["priority"]),
                    now,
                ),
            )
            inserted += cursor.rowcount
    after = plan(connection, contract)
    return {
        **after,
        "inserted": inserted,
        "before_to_insert": before["to_insert"],
        "applied": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    connection = sqlite3.connect(args.database, timeout=60)
    try:
        connection.execute("PRAGMA busy_timeout=60000")
        result = (
            apply(connection, contract)
            if args.apply
            else {**plan(connection, contract), "applied": False}
        )
    finally:
        connection.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
