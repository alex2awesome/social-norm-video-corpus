"""Expand / mutate the query queue based on observed hit rates.

Called by the loop after every `loop.eval_every` videos. Two behaviors:

  Exploit -- for high-hit-rate queries, generate close variations (swap a word,
             append a channel/keyword pulled from successful videos).
  Explore -- add fresh combinatorial queries from the {adjective x context x
             format} vocabulary, biased toward templates that have paid off.

A Thompson-sampling-style score (Beta(hits+1, misses+1) sampled deterministically
via the per-query rate plus a uniform-ish jitter derived from a rolling counter)
ranks which templates to draw from. We avoid Math.random/time-based seeds so the
process stays reproducible/resumable; jitter comes from a counter persisted via
the DB row count.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from . import state
from .query_scope import audit_query_scope

log = logging.getLogger("querygen")

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with",
    "compilation", "reaction", "caught", "camera", "video", "shorts", "clip",
    "clips", "part", "full", "official", "youtube", "vs", "ft", "feat",
}


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
            if w not in _STOP]


def _high_hit_queries(conn, platform: str, min_processed: int,
                      min_rate: float) -> list[dict]:
    rows = conn.execute(
        """SELECT query, videos_processed, videos_with_hits,
                  1.0*videos_with_hits/videos_processed AS rate
           FROM queries
           WHERE platform = ? AND videos_processed >= ? AND active = 1
           ORDER BY rate DESC""",
        (platform, min_processed),
    ).fetchall()
    return [dict(r) for r in rows if r["rate"] >= min_rate]


def _hit_video_terms(conn, platform: str, limit: int = 40) -> tuple[list[str], list[str]]:
    """Frequent title keywords + channel names from this platform's hit videos."""
    rows = conn.execute(
        """SELECT title, channel FROM seen_videos
           WHERE is_hit=1 AND platform=? ORDER BY processed_at DESC LIMIT ?""",
        (platform, limit),
    ).fetchall()
    kw: dict[str, int] = {}
    channels: dict[str, int] = {}
    for r in rows:
        for t in _tokens(r["title"]):
            kw[t] = kw.get(t, 0) + 1
        if r["channel"]:
            channels[r["channel"]] = channels.get(r["channel"], 0) + 1
    top_kw = [k for k, _ in sorted(kw.items(), key=lambda x: -x[1])[:15]]
    top_ch = [c for c, _ in sorted(channels.items(), key=lambda x: -x[1])[:8]]
    return top_kw, top_ch


def _combos(vocab: dict, pick: int, jitter: int) -> list[str]:
    """Deterministic spread of {adjective x context x format} combos.

    `jitter` (a rolling counter) rotates the starting offset so successive calls
    surface different combinations without RNG.
    """
    adj = vocab.get("adjective", [])
    ctx = vocab.get("context", [])
    fmt = vocab.get("format", [])
    if not (adj and ctx and fmt):
        return []
    out = []
    n = max(len(adj), len(ctx), len(fmt))
    for i in range(pick):
        j = jitter + i
        a = adj[(j) % len(adj)]
        c = ctx[(j * 2 + 1) % len(ctx)]
        f = fmt[(j * 3 + 2) % len(fmt)]
        out.append(f"{a} {c} {f}")
    return out


def evaluate_and_expand(conn, cfg: dict, vocab: dict) -> dict:
    """Run one exploit+explore round per text-query platform.

    Only free-text platforms (youtube/dailymotion/odysee) are expanded; Reddit
    targets are fixed subreddit names and left as-is.
    """
    from .sources import TEXT_QUERY_PLATFORMS

    loop_cfg = cfg["loop"]
    min_rate = loop_cfg.get("min_hit_rate", 0.05)
    cap = loop_cfg.get("max_videos_per_query", 50)
    max_queue = loop_cfg.get("max_queue_size", 5000)

    # --- prune underperformers past the cap (all platforms) ----------------
    pruned = conn.execute(
        """UPDATE queries SET active=0
           WHERE active=1 AND videos_processed >= ?
             AND 1.0*videos_with_hits/videos_processed < ?""",
        (cap, min_rate),
    ).rowcount
    conn.commit()

    active_count = conn.execute(
        "SELECT COUNT(*) AS n FROM queries WHERE active=1").fetchone()["n"]
    if active_count >= max_queue:
        log.info("queue at cap (%d); skipping expansion", active_count)
        return {"pruned": pruned, "added_exploit": 0, "added_explore": 0}

    jitter = conn.execute("SELECT COUNT(*) AS n FROM queries").fetchone()["n"]

    # which text-query platforms are actually in play?
    platforms = [r["platform"] for r in conn.execute(
        "SELECT DISTINCT platform FROM queries WHERE active=1").fetchall()
        if r["platform"] in TEXT_QUERY_PLATFORMS]

    added_exploit = added_explore = 0
    for platform in platforms:
        # exploit: append winning-video keywords to that platform's winners
        winners = _high_hit_queries(conn, platform, min_processed=5,
                                    min_rate=max(min_rate, 0.1))
        kw, channels = _hit_video_terms(conn, platform)
        for w in winners[:5]:
            for extra in (kw[:3] + channels[:2]):
                cand = f"{w['query']} {extra}".strip()
                # BREADTH-FIRST (2026-06-04): exploit is depth-drilling (append a
                # winning video's title keywords to a winning query) -> it re-fetches
                # the same neighborhood (e.g. the "What Would You Do?" cluster looped
                # YouTube, starving the never-run AV/instructional seeds). Priority is
                # 0.3 (idle-fill) so the curated seed/taxonomy/instructional terms
                # sweep round-robin (LRU) FIRST; exploit only runs once breadth is
                # exhausted. Was 1.5 (which strictly outranked seeds at 1.0).
                decision = audit_query_scope(cand, "exploit")
                inserted = decision.allowed and state.add_query(
                    conn, platform, cand, "exploit", priority=0.3)
                state.record_query_proposal(
                    conn, platform=platform, query=cand, source="exploit",
                    decision=decision, inserted=inserted)
                if inserted:
                    added_exploit += 1
        # explore: combinatorial vocab spread for that platform
        for cand in _combos(vocab, pick=6, jitter=jitter):
            decision = audit_query_scope(cand, "explore")
            inserted = decision.allowed and state.add_query(
                conn, platform, cand, "explore", priority=1.0)
            state.record_query_proposal(
                conn, platform=platform, query=cand, source="explore",
                decision=decision, inserted=inserted)
            if inserted:
                added_explore += 1

    summary = {"pruned": pruned, "added_exploit": added_exploit,
               "added_explore": added_explore, "active_after": active_count}
    log.info("query gen: %s", summary)
    return summary
