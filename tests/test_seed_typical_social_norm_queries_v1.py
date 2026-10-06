import copy
import time
from pathlib import Path

import pytest

from scripts.seed_typical_social_norm_queries_v1 import seed, validate
from src import state


def cfg(tmp_path: Path):
    return {
        "paths": {"state_db": str(tmp_path / "state.db")},
        "scheduler": {"blocked_query_patterns": [r"\bpolice\b", r"\bcops?\b"]},
    }


def plan():
    return {
        "source": "typical_social_norm_v1",
        "policy": {"police_or_law_enforcement_confrontation_allowed": False},
        "queries": [
            {
                "query": "roommate chore conflict role play",
                "pillar": "instructional",
                "category": "instr_typical_home",
                "platforms": ["youtube", "dailymotion"],
                "canary": True,
            },
            {
                "query": "line cutting incident video explained",
                "pillar": "commentary",
                "category": "comm_typical_queue",
                "platforms": ["youtube"],
                "canary": False,
            },
        ],
    }


def audit():
    rows = []
    for query, pillar, decision in (
        ("roommate chore conflict role play", "instructional", "canary_approved"),
        ("line cutting incident video explained", "commentary", "held_pending_canary_output_audit"),
    ):
        rows.append({
            "query": query,
            "pillar": pillar,
            "manual_reviewed": True,
            "ordinary_social_norm": "yes",
            "police_confrontation": "no",
            "specific_visible_target": "yes",
            "exact_prior_query": "no",
            "decision": decision,
            "risk": "Requires manual output review.",
        })
    return rows


def analysis():
    return {"proposal": {
        "queries": 2,
        "exact_prior_query_count": 0,
        "blocked_pattern_match_count": 0,
    }}


def test_seeds_only_canaries_active_and_holds_remaining_queries(tmp_path: Path):
    settings = cfg(tmp_path)
    conn = state.init_db(settings)
    result = seed(conn, plan(), audit(), analysis(), settings)
    assert result["inserted_query_rows"] == 3
    rows = conn.execute("SELECT query,active,family FROM queries ORDER BY query,platform").fetchall()
    active = {(r["query"], r["family"]): r["active"] for r in rows}
    assert active[("roommate chore conflict role play", "instructional")] == 1
    assert active[("line cutting incident video explained", "commentary")] == 0


def test_rejects_blocked_or_unreviewed_query(tmp_path: Path):
    settings = cfg(tmp_path)
    bad = plan()
    bad["queries"][0]["query"] = "police confrontation role play"
    bad_audit = audit()
    bad_audit[0]["query"] = "police confrontation role play"
    with pytest.raises(ValueError, match="blocked"):
        validate(bad, bad_audit, analysis(), settings)
    missing = audit()[:1]
    with pytest.raises(ValueError, match="exactly cover"):
        validate(plan(), missing, analysis(), settings)
