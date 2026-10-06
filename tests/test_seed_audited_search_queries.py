import sqlite3
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.seed_audited_search_queries import apply, load_contract, plan


def database():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """CREATE TABLE queries (
            platform TEXT NOT NULL,
            query TEXT NOT NULL,
            priority REAL,
            active INTEGER,
            source TEXT,
            category TEXT,
            added_at REAL,
            PRIMARY KEY (platform, query)
        )"""
    )
    return connection


def contract():
    return {
        "version": "v1",
        "active": True,
        "platform": "dailymotion",
        "priority": 3.25,
        "source": "manual_audit_search_v1",
        "queries": [
            {
                "pillar": "instructional",
                "category": "instr_scene_audited_v1",
                "query": "acted social scene",
                "reviewed": 3,
                "strict_target_passes": 2,
            },
            {
                "pillar": "commentary",
                "category": "comm_scene_audited_v1",
                "query": "commentary with footage",
                "reviewed": 2,
                "strict_target_passes": 1,
            },
        ],
    }


def test_apply_is_idempotent_and_preserves_provenance():
    connection = database()
    first = apply(connection, contract())
    second = apply(connection, contract())
    assert first["inserted"] == 2
    assert second["inserted"] == 0
    rows = connection.execute(
        "SELECT query, source, category, priority, active FROM queries ORDER BY query"
    ).fetchall()
    assert rows == [
        (
            "acted social scene",
            "manual_audit_search_v1",
            "instr_scene_audited_v1",
            3.25,
            1,
        ),
        (
            "commentary with footage",
            "manual_audit_search_v1",
            "comm_scene_audited_v1",
            3.25,
            1,
        ),
    ]


def test_plan_does_not_mutate_database():
    connection = database()
    result = plan(connection, contract())
    assert len(result["to_insert"]) == 2
    assert connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 0


def test_contract_rejects_zero_pass_query(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
version: v1
active: true
platform: dailymotion
priority: 3.25
source: audit
queries:
  - pillar: instructional
    category: instr_scene
    query: unsupported
    reviewed: 2
    strict_target_passes: 0
"""
    )
    with pytest.raises(ValueError, match="invalid manual audit counts"):
        load_contract(path)


def test_contract_fails_closed_when_frozen_audit_hash_changes(tmp_path):
    root = tmp_path / "repo"
    config_dir = root / "config"
    audit_dir = root / "audit"
    config_dir.mkdir(parents=True)
    audit_dir.mkdir()
    manual = audit_dir / "manual.jsonl"
    selection = audit_dir / "selection.json"
    manual.write_text("changed\n")
    selection.write_text('{"items": []}\n')
    path = config_dir / "contract.yaml"
    path.write_text(
        """
version: v1
active: true
platform: dailymotion
priority: 3.25
source: audit
audit:
  validation_required_before_seed: true
  source: audit/manual.jsonl
  source_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  selection: audit/selection.json
  selection_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
queries:
  - pillar: instructional
    category: instr_scene
    query: acted scene
    reviewed: 1
    strict_target_passes: 1
"""
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_contract(path)


def test_production_contract_matches_frozen_manual_search_audit():
    root = Path(__file__).resolve().parents[1]
    frozen = root / "audit_runs" / "20260723_search_shadow_v4_video_audit"
    selection = json.loads((frozen / "selection.json").read_text())["items"]
    reviews = {
        row["uid"]: row
        for row in (
            json.loads(line)
            for line in (frozen / "manual_review.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    observed = Counter()
    for selected in selection:
        review = reviews[selected["uid"]]
        key = (selected["pillar"], selected["query"])
        observed[(key, "reviewed")] += 1
        observed[(key, "strict")] += int(review["strict_target_pass"])

    production = load_contract(root / "config" / "audited_search_queries_v1.yaml")
    selected_keys = set()
    for row in production["queries"]:
        key = (row["pillar"], row["query"])
        selected_keys.add(key)
        assert row["reviewed"] == observed[(key, "reviewed")]
        assert row["strict_target_passes"] == observed[(key, "strict")]
        assert observed[(key, "strict")] > 0

    all_supported = {
        (selected["pillar"], selected["query"])
        for selected in selection
        if selected["pillar"] in {"instructional", "commentary"}
        if observed[((selected["pillar"], selected["query"]), "strict")] > 0
    }
    assert selected_keys == all_supported
    assert not any(pillar == "witnessed" for pillar, _ in selected_keys)
    assert sum(row["reviewed"] for row in production["queries"]) == 29
    assert sum(row["strict_target_passes"] for row in production["queries"]) == 16
