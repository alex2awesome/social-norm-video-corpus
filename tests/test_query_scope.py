import sqlite3

from src import state
from src.query_scope import POLICY_VERSION, audit_query_scope


def test_observed_bad_trajectories_are_blocked() -> None:
    cases = {
        "cat vs snake": "nonhuman_spectacle",
        "animal fights in homes": "nonhuman_spectacle",
        "pet snake attacks cat": "nonhuman_spectacle",
        "bus accident caught on camera": "accident_or_disaster_spectacle",
        "school shooting caught on camera": "weapon_or_mass_violence",
        "Fortnite scammer reaction": "gameplay_or_virtual",
        "officer rescues shop customer": "authority_or_police",
        "CM Punk vs Christian": "sports_or_combat_entertainment",
        "digital marketing mistakes": "technical_or_business_topic",
    }
    for query, reason in cases.items():
        decision = audit_query_scope(query, "llm_expand")
        assert not decision.allowed
        assert decision.reason == reason
        assert decision.policy_version == POLICY_VERSION


def test_typical_ordinary_social_queries_survive() -> None:
    queries = [
        "customer berates cashier other shoppers intervene",
        "person cuts in line bystander calls them out",
        "passenger refuses seat for elderly rider reaction",
        "neighbor blocks shared driveway confrontation",
        "restaurant guest leaves huge mess staff reaction",
        "dog owner refuses to pick up waste neighbor confronts",
        "coworker takes credit for another person's work meeting",
        "roommate eats someone else's food argument",
        "wedding guest interrupts ceremony family reaction",
        "parent insults service worker child reacts",
    ]
    for query in queries:
        assert audit_query_scope(query, "llm_expand").allowed, query


def test_gate_does_not_change_curated_query_sources() -> None:
    decision = audit_query_scope("police bodycam shooting", "instructional")
    assert decision.allowed
    assert decision.reason == "not_automatic_expansion"


def test_query_proposal_audit_trail() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(state._SCHEMA)
    decision = audit_query_scope("cat vs snake", "llm_expand")
    state.record_query_proposal(
        conn, platform="dailymotion", query="cat vs snake",
        source="llm_expand", decision=decision, inserted=False,
        parent_video_id="dm__1", category="public")
    row = conn.execute("SELECT * FROM query_proposals").fetchone()
    assert row["allowed"] == 0
    assert row["reason"] == "nonhuman_spectacle"
    assert row["policy_version"] == POLICY_VERSION
    assert row["parent_video_id"] == "dm__1"

