#!/usr/bin/env python3
"""Replay scheduler selection over an exported query inventory without network I/O."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

import yaml

if __package__:
    from scripts.analyze_query_trajectory_v1 import read_trajectory
    from scripts.seed_typical_social_norm_queries_v1 import read_jsonl, seed
    from src import state
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.analyze_query_trajectory_v1 import read_trajectory
    from scripts.seed_typical_social_norm_queries_v1 import read_jsonl, seed
    from src import state


def simulate(
    trajectory,
    plan,
    audit,
    analysis,
    settings,
    *,
    picks: int,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="norm-query-scheduler-v2-") as tmp:
        cfg = {
            "paths": {"state_db": str(Path(tmp) / "state.db")},
            "scheduler": settings["scheduler"],
        }
        conn = state.init_db(cfg)
        conn.executemany(
            """INSERT OR IGNORE INTO queries
               (platform,query,priority,active,source,category,cursor,
                videos_processed,videos_with_hits,total_reaction_count,
                added_at,last_used,family)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    row["platform"], row["query"], row.get("priority", 1.0),
                    row.get("active", 1), row.get("source"), row.get("category"),
                    row.get("cursor"), row.get("videos_processed", 0),
                    row.get("videos_with_hits", 0), row.get("total_reaction_count", 0),
                    row.get("added_at"), row.get("last_used"),
                    state.infer_query_family(row.get("source"), row.get("category")),
                )
                for row in trajectory
            ],
        )
        conn.commit()
        state._apply_query_policy(conn, cfg)
        seed_report = seed(conn, plan, audit, analysis, cfg)
        excluded = conn.execute(
            "SELECT count(*) FROM queries WHERE policy_excluded=1"
        ).fetchone()[0]

        selected = []
        clock = time.time()
        for offset in range(picks):
            row = state.next_query(conn, cfg)
            if row is None:
                break
            policy = state.query_policy_reason(row["query"], cfg)
            if policy:
                raise AssertionError(f"scheduler selected blocked query: {row['query']}")
            selected.append(row)
            # The simulation intentionally assumes no new results. It tests
            # allocation, exclusion, and repeat suppression—not future yield.
            state.record_query_run(
                conn, row["platform"], row["query"], started_at=clock + offset - 1,
                candidates_returned=15, new_candidates=0,
                enqueued_candidates=0, skipped_candidates=0,
                cursor_before=row.get("cursor"), cursor_after=row.get("cursor"),
                cfg=cfg, now=clock + offset,
            )
        keys = [(r["platform"], r["query"]) for r in selected]
        canary_queries = {r["query"] for r in plan["queries"] if r["canary"]}
        report = {
            "kind": "query_scheduler_v2_exported_trajectory_simulation",
            "requested_picks": picks,
            "completed_picks": len(selected),
            "unique_selected_query_rows": len(set(keys)),
            "repeated_selection_count": len(keys) - len(set(keys)),
            "selected_blocked_theme_queries": 0,
            "policy_excluded_query_rows": excluded,
            "by_family": dict(sorted(Counter(r["family"] for r in selected).items())),
            "by_platform": dict(sorted(Counter(r["platform"] for r in selected).items())),
            "selected_canary_query_rows": sum(r["query"] in canary_queries for r in selected),
            "seed_report": seed_report,
            "assumption": "all simulated passes return already-seen results",
            "yield_estimated": False,
            "network_requests_made": 0,
            "automatic_acceptance": False,
            "corpus_mutation_authorized": False,
        }
        conn.close()
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--trajectory-analysis", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--picks", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(simulate(
        read_trajectory(args.trajectory),
        yaml.safe_load(args.plan.read_text()),
        read_jsonl(args.manual_audit),
        json.loads(args.trajectory_analysis.read_text()),
        yaml.safe_load(args.settings.read_text()),
        picks=args.picks,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
