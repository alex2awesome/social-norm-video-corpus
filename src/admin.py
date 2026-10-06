"""Admin/observability for the overnight crawl. Used by the periodic check-in.

  python -m src.admin --status                 # JSON health + top/bottom queries + sample norms
  python -m src.admin --add "kw one;kw two"    # add keywords (dailymotion+odysee), ; separated
"""
from __future__ import annotations

import argparse
import json

from . import state


def status(conn) -> dict:
    totals = conn.execute(
        """SELECT COUNT(*) seen, SUM(status='done') done, SUM(is_hit) hits,
                  COALESCE(SUM(n_reactions),0) reactions
           FROM seen_videos""").fetchone()
    by_source = [dict(r) for r in conn.execute(
        """SELECT source, COUNT(*) seen, SUM(status='done') done, SUM(is_hit) hits
           FROM seen_videos GROUP BY source ORDER BY hits DESC""")]
    tiers = {r["tier"]: r["n"] for r in conn.execute(
        "SELECT tier, COUNT(*) n FROM reactions GROUP BY tier")}
    top_q = [dict(r) for r in conn.execute(
        """SELECT platform, query, videos_processed vp, videos_with_hits vh,
                  1.0*videos_with_hits/videos_processed rate
           FROM queries WHERE videos_processed>=3
           ORDER BY rate DESC, vp DESC LIMIT 15""")]
    low_q = [dict(r) for r in conn.execute(
        """SELECT platform, query, videos_processed vp, videos_with_hits vh
           FROM queries WHERE videos_processed>=6 AND videos_with_hits=0
           ORDER BY vp DESC LIMIT 15""")]
    # sample of LLM-detected norms (the 'norm' is stored in context as "[norm] ...")
    norms = [r["context"] for r in conn.execute(
        "SELECT context FROM reactions WHERE tier=5 AND context LIKE '[%' ORDER BY id DESC LIMIT 30")]
    norms = [c.split("]")[0][1:] for c in norms if c]
    active = {r["platform"]: r["n"] for r in conn.execute(
        "SELECT platform, COUNT(*) n FROM queries WHERE active=1 GROUP BY platform")}
    modality = {r["modality"] or "miss/none": r["n"] for r in conn.execute(
        "SELECT modality, COUNT(*) n FROM seen_videos WHERE status='done' GROUP BY modality")}
    agents = {r["agent"]: r["n"] for r in conn.execute(
        "SELECT agent, COUNT(*) n FROM seen_videos WHERE is_hit=1 OR modality='commentary' GROUP BY agent")}
    return {
        "totals": dict(totals),
        "modality": modality,           # witnessed (video) vs commentary (text corpus)
        "agent": agents,                # human vs non_human (autonomous vehicle / robot)
        "reaction_tiers": tiers,         # tier 5 = LLM, others = keyword
        "by_source": by_source,
        "active_queries": active,
        "top_queries": top_q,
        "zero_hit_queries": low_q,
        "recent_llm_norms": norms,
    }


def add_keywords(conn, kws: list[str]) -> int:
    added = 0
    for q in kws:
        q = q.strip()
        if not q:
            continue
        added += state.add_query(conn, "dailymotion", q, "claude", priority=1.3)
        added += state.add_query(conn, "odysee", q, "claude", priority=0.5)
    conn.commit()
    return added


def add_subreddits(conn, subs: list[str]) -> int:
    added = 0
    for s in subs:
        s = s.strip().lstrip("r/")
        if s:
            added += state.add_query(conn, "reddit", s, "claude", priority=1.1)
    conn.commit()
    return added


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--add", help="semicolon-separated keywords for dailymotion+odysee")
    ap.add_argument("--add-subreddits", help="semicolon-separated subreddit names")
    args = ap.parse_args()
    conn = state.init_db()
    if args.add:
        print("added keywords:", add_keywords(conn, args.add.split(";")))
    if args.add_subreddits:
        print("added subreddits:", add_subreddits(conn, args.add_subreddits.split(";")))
    if args.status or not (args.add or args.add_subreddits):
        print(json.dumps(status(conn), indent=2, default=str))
