import copy
import sqlite3
from pathlib import Path

import pytest

from scripts.apply_audited_search_query_maintenance_v2 import apply, load_contract, plan


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config" / "audited_search_query_maintenance_v2.yaml"


def database(contract):
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """CREATE TABLE queries (
            platform TEXT NOT NULL, query TEXT NOT NULL, priority REAL,
            active INTEGER, source TEXT, category TEXT, added_at REAL,
            PRIMARY KEY (platform, query))"""
    )
    for row in contract["decisions"]:
        connection.execute(
            "INSERT INTO queries VALUES (?, ?, 3.25, 1, 'audit', 'category', 0)",
            (contract["platform"], row["query"]),
        )
    return connection


def test_production_contract_recomputes_all_128_manual_judgments():
    contract = load_contract(CONTRACT_PATH)
    report = contract["_validated_report"]
    assert report["overall"]["units"] == 128
    assert report["overall"]["sources"] == 44
    assert sum(row["decision"] == "pause" for row in contract["decisions"]) == 6
    assert sum(row["decision"] == "preserve_state" for row in contract["decisions"]) == 5


def test_dry_run_lists_exact_changes_without_mutation():
    contract = load_contract(CONTRACT_PATH)
    connection = database(contract)
    result = plan(connection, contract)
    assert len(result["active_queries_to_pause"]) == 6
    assert connection.execute("SELECT SUM(active) FROM queries").fetchone()[0] == 11


def test_apply_only_flips_zero_yield_active_flags_and_is_idempotent():
    contract = load_contract(CONTRACT_PATH)
    connection = database(contract)
    positive = {
        row["query"] for row in contract["decisions"] if row["decision"] == "preserve_state"
    }
    before = {
        row[0]: row[1:]
        for row in connection.execute("SELECT query, priority, source, category FROM queries")
    }
    first = apply(connection, contract)
    second = apply(connection, contract)
    assert first["updated_active_flags"] == 6
    assert second["updated_active_flags"] == 0
    rows = connection.execute(
        "SELECT query, active, priority, source, category FROM queries"
    ).fetchall()
    assert all(bool(active) == (query in positive) for query, active, *_ in rows)
    assert {
        query: (priority, source, category) for query, _, priority, source, category in rows
    } == before
    assert connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 11


def test_validator_rejects_pausing_a_positive_yield_query():
    contract = load_contract(CONTRACT_PATH)
    bad = copy.deepcopy(contract)
    bad.pop("_validated_report", None)
    positive = next(row for row in bad["decisions"] if row["decision"] == "preserve_state")
    positive["decision"] = "pause"
    # Exercise the invariant directly through a temporary contract with valid relative paths.
    import yaml
    temporary = CONTRACT_PATH.parent / "_temporary_bad_maintenance_contract.yaml"
    try:
        temporary.write_text(yaml.safe_dump(bad, sort_keys=False))
        with pytest.raises(ValueError, match="decision contradicts"):
            load_contract(temporary)
    finally:
        temporary.unlink(missing_ok=True)
