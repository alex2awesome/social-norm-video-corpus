#!/usr/bin/env python3
"""Reversibly quarantine failed-transfer Dailymotion related queries."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3


POLICY_REASON = "related_transfer_failed:20260808_v1"


def apply(conn: sqlite3.Connection, mode: str) -> dict:
    conn.row_factory = sqlite3.Row
    before = [dict(x) for x in conn.execute(
        """SELECT COALESCE(policy_reason,'') reason,count(*) n
           FROM queries WHERE platform='dmrelated' AND source='related'
           GROUP BY COALESCE(policy_reason,'') ORDER BY n DESC"""
    ).fetchall()]
    if mode == "quarantine":
        changed = conn.execute(
            """UPDATE queries SET policy_excluded=1,policy_reason=?
               WHERE platform='dmrelated' AND source='related'
                 AND COALESCE(policy_excluded,0)=0""", (POLICY_REASON,)
        ).rowcount
    elif mode == "restore":
        changed = conn.execute(
            """UPDATE queries SET policy_excluded=0,policy_reason=NULL
               WHERE platform='dmrelated' AND source='related'
                 AND policy_reason=?""", (POLICY_REASON,)
        ).rowcount
    else:
        raise ValueError(f"unknown mode: {mode}")
    conn.commit()
    after = conn.execute(
        """SELECT count(*) FROM queries WHERE platform='dmrelated'
           AND source='related' AND COALESCE(policy_excluded,0)=1"""
    ).fetchone()[0]
    return {"policy": POLICY_REASON, "mode": mode, "changed": changed,
            "related_queries": sum(x["n"] for x in before),
            "excluded_after": after, "before_by_reason": before,
            "queries_deleted": 0, "media_deleted": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--mode", choices=("quarantine", "restore"), default="quarantine")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db, timeout=60)
    print(json.dumps(apply(conn, args.mode), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
