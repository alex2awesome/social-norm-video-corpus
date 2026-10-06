#!/usr/bin/env python3
"""Report conservative audit coverage for the inactive weak-supervision policy."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import yaml


def scalar(conn: sqlite3.Connection, sql: str, params=()) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


def grouped(conn: sqlite3.Connection, sql: str, params=()) -> dict[str, int]:
    return {str(key): int(count) for key, count in conn.execute(sql, params)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/visual_audit/audit.db"))
    parser.add_argument(
        "--policy", type=Path, default=Path("config/weak_supervision_shadow_v1.yaml")
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    policy = yaml.safe_load(args.policy.read_text())
    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys=ON")

    totals = {
        "instructional_with_clip": scalar(
            conn, "SELECT COUNT(*) FROM items WHERE pillar='instructional' AND present=1 AND has_clip=1"
        ),
        "instructional_without_clip": scalar(
            conn, "SELECT COUNT(*) FROM items WHERE pillar='instructional' AND present=1 AND has_clip=0"
        ),
        "witnessed": scalar(
            conn, "SELECT COUNT(*) FROM items WHERE pillar='witnessed' AND present=1 AND has_clip=1"
        ),
        "commentary": scalar(
            conn, "SELECT COUNT(*) FROM items WHERE pillar='commentary' AND present=1"
        ),
    }
    reviewed = {
        "instructional_v4": scalar(
            conn,
            """SELECT COUNT(DISTINCT j.item_id)
               FROM judgments j JOIN items i ON i.item_id=j.item_id
               WHERE j.rubric_version='instructional_v4'
                 AND i.pillar='instructional'""",
        ),
        "witnessed_reroute_instructional_v4": scalar(
            conn,
            """SELECT COUNT(DISTINCT j.item_id)
               FROM judgments j JOIN items i ON i.item_id=j.item_id
               WHERE j.rubric_version='instructional_v4'
                 AND i.pillar='witnessed'""",
        ),
        "instructional_rendered_repairs": scalar(
            conn, "SELECT COUNT(*) FROM instructional_repair_judgments"
        ),
        "witnessed_v1": scalar(
            conn, "SELECT COUNT(DISTINCT item_id) FROM witnessed_judgments WHERE rubric_version='witnessed_v1'"
        ),
        "commentary_v2": scalar(
            conn, "SELECT COUNT(DISTINCT item_id) FROM commentary_judgments WHERE rubric_version='commentary_v2'"
        ),
    }
    decisions = {
        "instructional_v4": grouped(
            conn,
            """SELECT j.decision,COUNT(*)
               FROM judgments j JOIN items i ON i.item_id=j.item_id
               WHERE j.rubric_version='instructional_v4'
                 AND i.pillar='instructional'
               GROUP BY j.decision""",
        ),
        "witnessed_reroute_instructional_v4": grouped(
            conn,
            """SELECT j.decision,COUNT(*)
               FROM judgments j JOIN items i ON i.item_id=j.item_id
               WHERE j.rubric_version='instructional_v4'
                 AND i.pillar='witnessed'
               GROUP BY j.decision""",
        ),
        "instructional_rendered_repairs": grouped(
            conn, "SELECT decision,COUNT(*) FROM instructional_repair_judgments GROUP BY decision"
        ),
        "witnessed_v1": grouped(
            conn,
            "SELECT decision,COUNT(*) FROM witnessed_judgments WHERE rubric_version='witnessed_v1' GROUP BY decision",
        ),
        "commentary_v2": grouped(
            conn,
            "SELECT decision,COUNT(*) FROM commentary_judgments WHERE rubric_version='commentary_v2' GROUP BY decision",
        ),
    }
    report = {
        "policy_version": policy["version"],
        "policy_active": bool(policy["active"]),
        "conservative_potentially_affected_outputs": totals,
        "reviewed": reviewed,
        "decisions": decisions,
        "coverage": {
            "instructional_with_clip": reviewed["instructional_v4"] / totals["instructional_with_clip"],
            "witnessed": reviewed["witnessed_v1"] / totals["witnessed"],
            "commentary": reviewed["commentary_v2"] / totals["commentary"],
        },
        "intended_closed_vlm_executed": False,
        "changed_output_audit_complete": False,
        "ready_for_promotion": False,
        "blocking_gates": [
            "approved GPT-5.6-sol endpoint unavailable",
            "potentially affected corpus is not fully judged",
            "full frozen symmetric diff has not been produced",
            "every changed output has not been manually reviewed",
        ],
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end="")
    conn.close()


if __name__ == "__main__":
    main()
