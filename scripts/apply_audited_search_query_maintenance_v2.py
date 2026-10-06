#!/usr/bin/env python3
"""Reversibly pause search queries with zero dense-audit end-to-end yield.

The contract is validated from the raw frozen blind and post-reveal ledgers.
Only the ``active`` flag of exact zero-yield query rows can change. Query rows,
downloaded videos, priorities, labels, and positive-yield query state are
preserved. The CLI is dry-run unless ``--apply`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.evaluate_fresh_scene_query_audit import evaluate, read_jsonl
except ModuleNotFoundError:  # pragma: no cover - direct CLI execution
    from evaluate_fresh_scene_query_audit import evaluate, read_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_path(root: Path, value: str, digest: str) -> Path:
    path = root / value
    if not path.is_file():
        raise ValueError(f"missing frozen artifact: {value}")
    if len(digest) != 64 or sha256(path) != digest:
        raise ValueError(f"frozen artifact hash mismatch: {value}")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text())
    if not isinstance(contract, dict) or contract.get("active") is not True:
        raise ValueError("maintenance contract is absent or inactive")
    policy = contract.get("policy") or {}
    required_true = {
        "retrieval_only",
        "exact_query_match_only",
        "preserve_query_rows",
        "preserve_existing_videos",
        "preserve_positive_yield_query_state",
        "reversible_by_active_flag",
        "query_text_is_never_a_label",
    }
    if policy.get("action") != "pause_zero_end_to_end_yield_only" or any(
        policy.get(name) is not True for name in required_true
    ):
        raise ValueError("unsafe maintenance policy")

    root = path.resolve().parent.parent
    base_spec = contract.get("base_contract") or {}
    base_path = _verified_path(root, base_spec["path"], base_spec["sha256"])
    base = yaml.safe_load(base_path.read_text())
    base_queries = {str(row["query"]): row for row in base["queries"]}

    audit = contract.get("dense_manual_audit") or {}
    sealed_path = _verified_path(root, audit["sealed"], audit["sealed_sha256"])
    blind_path = _verified_path(root, audit["blind"], audit["blind_sha256"])
    post_path = _verified_path(root, audit["post"], audit["post_sha256"])
    report = evaluate(
        read_jsonl(sealed_path), read_jsonl(blind_path), read_jsonl(post_path)
    )
    if report["overall"]["units"] != audit["manually_reviewed_units"]:
        raise ValueError("manual unit count mismatch")
    if report["overall"]["sources"] != audit["source_videos"]:
        raise ValueError("manual source count mismatch")

    decisions = contract.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("maintenance decisions are empty")
    if {str(row.get("query")) for row in decisions} != set(base_queries):
        raise ValueError("maintenance decisions must exactly partition base queries")
    for row in decisions:
        query = str(row["query"])
        if row.get("decision") not in {"pause", "preserve_state"}:
            raise ValueError(f"invalid decision for {query}")
        observed = report["by_query"].get(query)
        if observed is None:
            raise ValueError(f"query absent from dense audit: {query}")
        expected = {
            "strict_pass_sources": observed["strict_pass_sources"],
            "reviewed_sources": observed["sources"],
            "strict_pass_units": observed["strict_pass_units"],
            "reviewed_units": observed["units"],
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"{query}: {field} mismatch")
        zero_yield = not observed["strict_pass_sources"] and not observed["strict_pass_units"]
        if (row["decision"] == "pause") != zero_yield:
            raise ValueError(f"{query}: decision contradicts dense manual audit")
    contract["_validated_report"] = report
    return contract


def plan(connection: sqlite3.Connection, contract: dict[str, Any]) -> dict[str, Any]:
    platform = str(contract["platform"])
    paused = [row["query"] for row in contract["decisions"] if row["decision"] == "pause"]
    preserved = [
        row["query"] for row in contract["decisions"] if row["decision"] == "preserve_state"
    ]
    states = {
        query: connection.execute(
            "SELECT active FROM queries WHERE platform=? AND query=?",
            (platform, query),
        ).fetchone()
        for query in paused + preserved
    }
    return {
        "version": contract["version"],
        "platform": platform,
        "paused_contract_queries": len(paused),
        "preserved_contract_queries": len(preserved),
        "active_queries_to_pause": sorted(
            query for query in paused if states[query] is not None and states[query][0] != 0
        ),
        "already_inactive": sorted(
            query for query in paused if states[query] is not None and states[query][0] == 0
        ),
        "missing_queries": sorted(query for query, state in states.items() if state is None),
        "positive_yield_states": {
            query: (None if states[query] is None else int(states[query][0]))
            for query in sorted(preserved)
        },
        "policy": "reversible_exact_query_active_flag_only",
        "videos_mutated": False,
        "rows_deleted": 0,
    }


def apply(connection: sqlite3.Connection, contract: dict[str, Any]) -> dict[str, Any]:
    before = plan(connection, contract)
    platform = str(contract["platform"])
    paused = [row["query"] for row in contract["decisions"] if row["decision"] == "pause"]
    with connection:
        updated = sum(
            connection.execute(
                "UPDATE queries SET active=0 WHERE platform=? AND query=? AND active<>0",
                (platform, query),
            ).rowcount
            for query in paused
        )
    return {
        **plan(connection, contract),
        "updated_active_flags": updated,
        "before_active_queries_to_pause": before["active_queries_to_pause"],
        "applied": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    connection = sqlite3.connect(args.database, timeout=60)
    try:
        connection.execute("PRAGMA busy_timeout=60000")
        result = apply(connection, contract) if args.apply else {
            **plan(connection, contract), "applied": False
        }
    finally:
        connection.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
