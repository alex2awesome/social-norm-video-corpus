import sqlite3

from src import state
from src.snowball_scope import POLICY_VERSION, audit_snowball_parent


def scene(**changes):
    base = {"scene_type": "action", "n_people": 3, "reactor_role": "bystander",
            "reaction_strength": 4}
    base.update(changes)
    return base


def test_strong_ordinary_bystander_parent_is_allowed() -> None:
    result = audit_snowball_parent(
        title="Student stops a bully and another classmate intervenes",
        agent="human", scene=scene(), reaction_count=1)
    assert result.allowed
    assert result.reason == "strong_in_scope_bystander_parent"
    assert result.policy_version == POLICY_VERSION


def test_observed_drift_families_are_blocked_even_with_positive_scene_metadata() -> None:
    cases = [
        ("Russian Road Rage and Accidents November", "road_accident_or_rage"),
        ("Cops arrest wanted man", "title_scope:authority_or_police"),
        ("CRAZY ROAD RAGE FIGHT PRANK", "road_accident_or_rage"),
        ("SML Movie Chef Pee Pees Clone", "produced_or_compilation"),
        ("Richest scammer in Fortnite save the world pve", "title_scope:gameplay_or_virtual"),
        ("Portugal stages mass strike in protest", "protest_or_spectacle"),
    ]
    for title, reason in cases:
        result = audit_snowball_parent(
            title=title, agent="human", scene=scene(), reaction_count=2)
        assert not result.allowed
        assert result.reason == reason


def test_role_and_strength_fail_closed() -> None:
    assert audit_snowball_parent(title="Customer berates cashier", agent="human",
        scene=scene(reactor_role="victim"), reaction_count=1).reason == "reactor_not_distinct_bystander"
    assert audit_snowball_parent(title="Neighbor dispute", agent="human",
        scene=scene(n_people=2), reaction_count=1).reason == "fewer_than_three_people"
    assert audit_snowball_parent(title="Shopper cuts line", agent="human",
        scene=scene(reaction_strength=2), reaction_count=1).reason == "weak_or_uncertain_reaction"
    assert audit_snowball_parent(title="Shopper cuts line", agent="human",
        scene={}, reaction_count=1).reason == "not_live_action"


def test_audit_trail_preserves_decision_evidence() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(state._SCHEMA)
    decision = audit_snowball_parent(
        title="Customer berates cashier and shopper intervenes", agent="human",
        scene=scene(), reaction_count=1)
    state.record_snowball_proposal(
        conn, parent_video_id="dailymotion__x", platform="dmrelated",
        related_query="x", category="wit_service", decision=decision,
        inserted=True)
    row = conn.execute("SELECT * FROM snowball_proposals").fetchone()
    assert row["allowed"] == 1
    assert row["policy_version"] == POLICY_VERSION
    assert '"reactor_role": "bystander"' in row["evidence_json"]
