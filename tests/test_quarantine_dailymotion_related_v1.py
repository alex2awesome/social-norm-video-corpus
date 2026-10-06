import sqlite3

from scripts.quarantine_dailymotion_related_v1 import POLICY_REASON, apply
from src import state


def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(state._SCHEMA)
    conn.executemany(
        "INSERT INTO queries(platform,query,source,policy_excluded,policy_reason) VALUES(?,?,?,?,?)",
        [("dmrelated", "x1", "related", 0, None),
         ("dmrelated", "x2", "related", 1, "other_policy"),
         ("dailymotion", "ordinary", "seed", 0, None)])
    return conn


def test_quarantine_and_restore_are_exact_and_reversible() -> None:
    conn = db()
    report = apply(conn, "quarantine")
    assert report["changed"] == 1
    assert conn.execute("SELECT policy_reason FROM queries WHERE query='x1'").fetchone()[0] == POLICY_REASON
    assert conn.execute("SELECT policy_reason FROM queries WHERE query='x2'").fetchone()[0] == "other_policy"
    restored = apply(conn, "restore")
    assert restored["changed"] == 1
    assert conn.execute("SELECT policy_excluded FROM queries WHERE query='x1'").fetchone()[0] == 0
    assert conn.execute("SELECT policy_excluded FROM queries WHERE query='x2'").fetchone()[0] == 1
