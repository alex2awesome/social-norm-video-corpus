"""State + config layer for the norm-scraper pipeline.

Centralizes config loading (config/settings.yaml) and the SQLite store used for
deduplication, per-(platform,query) hit-rate tracking, pagination cursors, and
the reaction-frequency report.

Schema
------
seen_videos : every candidate we have enumerated (status tracks progress).
              `video_id` holds the globally-unique uid ("{source}__{native}").
queries     : the work queue + per-target statistics. PK is (platform, query)
              so the same text can run on multiple platforms; `cursor` paginates.
reactions   : one row per matched reaction phrase (for the frequency report).

The DB is the single source of truth so the loop is resumable.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    return PROJECT_ROOT


_CONFIG_CACHE: Optional[dict] = None


def load_config(path: Optional[str] = None) -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE
    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "settings.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if path is None:
        _CONFIG_CACHE = cfg
    return cfg


def resolve_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dirs(cfg: dict) -> None:
    paths = cfg["paths"]
    for key in ("raw_video", "transcripts", "hits", "metadata", "logs"):
        resolve_path(paths[key]).mkdir(parents=True, exist_ok=True)
    resolve_path(paths.get("discussion", "data/discussion")).mkdir(parents=True, exist_ok=True)
    resolve_path(paths.get("negatives", "data/negatives")).mkdir(parents=True, exist_ok=True)
    resolve_path(paths.get("instructional", "data/instructional")).mkdir(parents=True, exist_ok=True)
    resolve_path(paths["state_db"]).parent.mkdir(parents=True, exist_ok=True)


def load_yaml(rel: str) -> Any:
    with open(PROJECT_ROOT / rel) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_videos (
    video_id      TEXT PRIMARY KEY,   -- uid: "{source}__{native_id}"
    source        TEXT,
    title         TEXT,
    channel       TEXT,
    duration      REAL,
    url           TEXT,
    query         TEXT,
    category      TEXT,               -- norm-taxonomy key the query belongs to (paper provenance)
    modality      TEXT,               -- witnessed (video corpus) | commentary (text corpus) | NULL
    agent         TEXT,               -- human | non_human (autonomous vehicle / robot) | NULL
    platform      TEXT,
    status        TEXT,               -- enumerated|downloaded|transcribed|done|skipped|error
    n_reactions   INTEGER DEFAULT 0,
    is_hit        INTEGER DEFAULT 0,
    error         TEXT,
    skip_reason   TEXT,
    enumerated_at REAL,
    processed_at  REAL
);

CREATE TABLE IF NOT EXISTS queries (
    platform             TEXT NOT NULL,
    query                TEXT NOT NULL,
    priority             REAL DEFAULT 1.0,
    active               INTEGER DEFAULT 1,
    source               TEXT,    -- seed|exploit|explore|title|channel|taxonomy|related
    category             TEXT,    -- norm-taxonomy key (propagated to snowball children)
    cursor               TEXT,    -- pagination cursor for the adapter
    videos_processed     INTEGER DEFAULT 0,
    videos_with_hits     INTEGER DEFAULT 0,
    total_reaction_count INTEGER DEFAULT 0,
    added_at             REAL,
    last_used            REAL,
    family               TEXT,
    zero_new_streak      INTEGER DEFAULT 0,
    cooldown_until       REAL,
    last_new_at          REAL,
    policy_excluded      INTEGER DEFAULT 0,
    policy_reason        TEXT,
    PRIMARY KEY (platform, query)
);

CREATE TABLE IF NOT EXISTS query_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    platform            TEXT NOT NULL,
    query               TEXT NOT NULL,
    family              TEXT,
    started_at          REAL,
    finished_at         REAL,
    run_status          TEXT,
    cursor_before       TEXT,
    cursor_after        TEXT,
    candidates_returned INTEGER DEFAULT 0,
    new_candidates      INTEGER DEFAULT 0,
    enqueued_candidates INTEGER DEFAULT 0,
    skipped_candidates  INTEGER DEFAULT 0,
    already_seen        INTEGER DEFAULT 0,
    zero_new_streak     INTEGER DEFAULT 0,
    cooldown_until      REAL
);

CREATE TABLE IF NOT EXISTS query_proposals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    proposed_at       REAL NOT NULL,
    platform          TEXT NOT NULL,
    query             TEXT NOT NULL,
    generation_source TEXT NOT NULL,
    parent_video_id   TEXT,
    category          TEXT,
    allowed           INTEGER NOT NULL,
    reason            TEXT NOT NULL,
    policy_version    TEXT NOT NULL,
    matched_text      TEXT,
    inserted          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS snowball_proposals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    proposed_at     REAL NOT NULL,
    parent_video_id TEXT NOT NULL,
    platform        TEXT NOT NULL,
    related_query   TEXT NOT NULL,
    category        TEXT,
    allowed         INTEGER NOT NULL,
    reason          TEXT NOT NULL,
    policy_version  TEXT NOT NULL,
    matched_text    TEXT,
    evidence_json   TEXT NOT NULL,
    inserted        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    TEXT,
    clip_idx    INTEGER,
    phrase      TEXT,
    tier        INTEGER,
    tag         TEXT,
    start_time  REAL,
    end_time    REAL,
    speaker_id  TEXT,
    context     TEXT
);

CREATE INDEX IF NOT EXISTS idx_reactions_phrase ON reactions(phrase);
CREATE INDEX IF NOT EXISTS idx_seen_status ON seen_videos(status);
CREATE INDEX IF NOT EXISTS idx_query_runs_finished ON query_runs(finished_at);
CREATE INDEX IF NOT EXISTS idx_query_proposals_time ON query_proposals(proposed_at);
CREATE INDEX IF NOT EXISTS idx_snowball_proposals_time ON snowball_proposals(proposed_at);
"""


def connect(cfg: Optional[dict] = None) -> sqlite3.Connection:
    cfg = cfg or load_config()
    db_path = resolve_path(cfg["paths"]["state_db"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=60000;")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, decl: str = "TEXT") -> None:
    """Idempotent ALTER ... ADD COLUMN so an existing DB picks up new columns."""
    have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if col not in have:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def init_db(cfg: Optional[dict] = None) -> sqlite3.Connection:
    cfg = cfg or load_config()
    conn = connect(cfg)
    conn.executescript(_SCHEMA)
    # migrate older DBs that predate the `category`/`modality` columns
    _ensure_column(conn, "seen_videos", "category")
    _ensure_column(conn, "seen_videos", "modality")
    _ensure_column(conn, "seen_videos", "agent")
    _ensure_column(conn, "seen_videos", "skip_reason")
    # query_source = the QUERY's provenance type (seed/taxonomy/instructional/
    # exploit/llm_expand/related/...), distinct from `source` (= platform adapter).
    # Lets every clip trace to WHICH KIND of query surfaced it (e.g. instructional YT).
    _ensure_column(conn, "seen_videos", "query_source")
    _ensure_column(conn, "queries", "category")
    _ensure_column(conn, "queries", "family")
    _ensure_column(conn, "queries", "zero_new_streak", "INTEGER DEFAULT 0")
    _ensure_column(conn, "queries", "cooldown_until", "REAL")
    _ensure_column(conn, "queries", "last_new_at", "REAL")
    _ensure_column(conn, "queries", "policy_excluded", "INTEGER DEFAULT 0")
    _ensure_column(conn, "queries", "policy_reason")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_queries_ready "
        "ON queries(active, policy_excluded, cooldown_until, family, platform)"
    )
    _backfill_query_families(conn)
    _apply_query_policy(conn, cfg)
    conn.commit()
    return conn


def infer_query_family(source: Optional[str], category: Optional[str]) -> str:
    """Map provenance/category to the collection pillar the query serves."""
    src = (source or "").casefold()
    cat = (category or "").casefold()
    if cat.startswith("instr_") or src == "instructional":
        return "instructional"
    if cat.startswith("comm_") or src == "commentary":
        return "commentary"
    if cat.startswith("null_") or src == "null":
        return "negative"
    if cat.startswith("wit_"):
        return "witnessed"
    # The legacy taxonomy, reaction-phrase, related, and LLM expansion veins
    # were all built to retrieve witnessed reaction candidates.
    return "witnessed"


def _backfill_query_families(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT platform, query, source, category FROM queries "
        "WHERE family IS NULL OR trim(family)=''"
    ).fetchall()
    conn.executemany(
        "UPDATE queries SET family=? WHERE platform=? AND query=?",
        [
            (infer_query_family(r["source"], r["category"]), r["platform"], r["query"])
            for r in rows
        ],
    )


def query_policy_reason(query: str, cfg: Optional[dict] = None) -> Optional[str]:
    """Return a reversible exclusion reason for newly disallowed query themes."""
    scheduler = ((cfg or load_config()).get("scheduler", {}) or {})
    for pattern in scheduler.get("blocked_query_patterns", []) or []:
        if re.search(str(pattern), query or "", flags=re.IGNORECASE):
            return f"blocked_theme:{pattern}"
    return None


def _apply_query_policy(conn: sqlite3.Connection, cfg: dict) -> None:
    """Refresh only config-owned exclusions; never deactivate or delete a query."""
    from .query_scope import AUTO_SOURCES, audit_query_scope

    rows = conn.execute(
        "SELECT platform, query, source, policy_reason FROM queries"
    ).fetchall()
    updates = []
    for row in rows:
        reason = query_policy_reason(row["query"], cfg)
        if not reason and row["source"] in AUTO_SOURCES:
            decision = audit_query_scope(row["query"], row["source"])
            if not decision.allowed:
                reason = f"auto_scope:{decision.policy_version}:{decision.reason}"
        old = row["policy_reason"] or ""
        if reason or old.startswith(("blocked_theme:", "auto_scope:")):
            updates.append((1 if reason else 0, reason, row["platform"], row["query"]))
    conn.executemany(
        "UPDATE queries SET policy_excluded=?, policy_reason=? "
        "WHERE platform=? AND query=?",
        updates,
    )


def seed_queries(conn: sqlite3.Connection, queries: Iterable[str],
                 platform: str = "youtube", source: str = "seed",
                 category: Optional[str] = None, priority: float = 1.0,
                 family: Optional[str] = None, active: bool = True,
                 cfg: Optional[dict] = None) -> int:
    """Insert seed queries for a platform if absent. Returns number newly added."""
    now = time.time()
    added = 0
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        if query_policy_reason(q, cfg):
            continue
        query_family = family or infer_query_family(source, category)
        cur = conn.execute(
            """INSERT OR IGNORE INTO queries
               (platform, query, source, category, priority, added_at, family, active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (platform, q, source, category, priority, now, query_family, int(active)),
        )
        added += cur.rowcount
    conn.commit()
    return added


# --- dedup / video lifecycle ------------------------------------------------

def is_seen(conn: sqlite3.Connection, uid: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM seen_videos WHERE video_id = ?", (uid,)
    ).fetchone() is not None


def enumerate_video(conn: sqlite3.Connection, cand: dict, query: str,
                    platform: str, category: Optional[str] = None,
                    query_source: Optional[str] = None) -> bool:
    """Record a freshly-enumerated candidate. Returns False if already seen.

    `category` is the norm-taxonomy key the surfacing query belongs to; it is
    stored per-video so the paper can trace each clip back to its norm class
    (including snowball children, which inherit the seed query's category).
    `query_source` is the surfacing query's PROVENANCE TYPE (seed/taxonomy/
    instructional/exploit/llm_expand/related/...), stored separately from `source`
    (= platform adapter) so each clip records which kind of query found it."""
    if is_seen(conn, cand["uid"]):
        return False
    # CONTENT dedup (uid dedup can't see re-uploads): the same video re-uploaded
    # under a different id ("Strangers in a Bed - Full Film" was saved 3x) has the
    # same title and ~the same duration. Record it as 'duplicate' so it is never
    # re-enumerated, but tell the caller to skip it.
    title = (cand.get("title") or "").strip().lower()
    dur = cand.get("duration")
    if title and dur is not None:
        row = conn.execute(
            """SELECT 1 FROM seen_videos
               WHERE lower(trim(title)) = ? AND duration IS NOT NULL
                 AND ABS(duration - ?) <= 2 LIMIT 1""",
            (title, dur)).fetchone()
        if row:
            conn.execute(
                """INSERT OR IGNORE INTO seen_videos
                   (video_id, source, title, channel, duration, url, query, category,
                    query_source, platform, status, skip_reason, enumerated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'duplicate',
                           'content_duplicate', ?)""",
                (cand["uid"], cand["source"], cand.get("title"), cand.get("channel"),
                 cand.get("duration"), cand.get("url"), query, category, query_source,
                 platform, time.time()))
            conn.commit()
            return False
    conn.execute(
        """INSERT OR IGNORE INTO seen_videos
           (video_id, source, title, channel, duration, url, query, category,
            query_source, platform, status, enumerated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'enumerated', ?)""",
        (cand["uid"], cand["source"], cand.get("title"), cand.get("channel"),
         cand.get("duration"), cand.get("url"), query, category, query_source,
         platform, time.time()),
    )
    conn.commit()
    return True


def set_status(conn: sqlite3.Connection, uid: str, status: str,
               error: Optional[str] = None,
               skip_reason: Optional[str] = None) -> None:
    conn.execute(
        "UPDATE seen_videos SET status=?, error=?, skip_reason=? WHERE video_id=?",
        (status, error, skip_reason, uid),
    )
    conn.commit()


def finalize_video(conn: sqlite3.Connection, uid: str, n_reactions: int,
                   modality: Optional[str] = None, agent: Optional[str] = None) -> None:
    conn.execute(
        """UPDATE seen_videos
           SET status='done', n_reactions=?, is_hit=?, modality=?, agent=?, processed_at=?
           WHERE video_id=?""",
        (n_reactions, 1 if n_reactions > 0 else 0, modality, agent, time.time(), uid),
    )
    conn.commit()


def finalize_discussion(conn: sqlite3.Connection, uid: str,
                        agent: Optional[str] = None) -> None:
    """A commentary hit: done, tagged for the text corpus, NOT a video hit."""
    conn.execute(
        """UPDATE seen_videos
           SET status='done', n_reactions=0, is_hit=0, modality='commentary',
               agent=?, processed_at=?
           WHERE video_id=?""",
        (agent, time.time(), uid),
    )
    conn.commit()


def record_reactions(conn: sqlite3.Connection, uid: str, matches: list[dict]) -> None:
    conn.executemany(
        """INSERT INTO reactions
           (video_id, clip_idx, phrase, tier, tag, start_time, end_time, speaker_id, context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(uid, m.get("clip_idx"), m["phrase"], m["tier"], m.get("tag"),
          m["start"], m["end"], m.get("speaker"), m.get("context")) for m in matches],
    )
    conn.commit()


def add_recovered_reactions(conn: sqlite3.Connection, uid: str,
                            matches: list[dict]) -> None:
    """Persist audio-recovered reactions to the DB so reports match metadata.json.

    audio_recover appends recovered reaction windows to the hit's metadata.json but
    (historically) never touched SQLite, so report()'s SUM(n_reactions) and the
    reactions table undercounted them. This inserts the rows and bumps
    seen_videos.n_reactions for the videos audio_recover enriched. The video is
    already a witnessed hit (it has a hits/ dir), so is_hit stays 1.
    """
    if not matches:
        return
    record_reactions(conn, uid, matches)
    conn.execute(
        "UPDATE seen_videos SET n_reactions = n_reactions + ?, is_hit = 1 "
        "WHERE video_id = ?",
        (len(matches), uid),
    )
    conn.commit()


# --- query stats / queue ----------------------------------------------------

def update_query_stats(conn: sqlite3.Connection, platform: str, query: str,
                       processed_delta: int, hit_delta: int,
                       reaction_delta: int) -> None:
    conn.execute(
        """UPDATE queries SET
             videos_processed = videos_processed + ?,
             videos_with_hits = videos_with_hits + ?,
             total_reaction_count = total_reaction_count + ?,
             last_used = ?
           WHERE platform = ? AND query = ?""",
        (processed_delta, hit_delta, reaction_delta, time.time(), platform, query),
    )
    conn.commit()


def set_cursor(conn: sqlite3.Connection, platform: str, query: str,
               cursor: Optional[str]) -> None:
    conn.execute("UPDATE queries SET cursor=? WHERE platform=? AND query=?",
                 (cursor, platform, query))
    conn.commit()


def deactivate_query(conn: sqlite3.Connection, platform: str, query: str) -> None:
    conn.execute("UPDATE queries SET active=0 WHERE platform=? AND query=?",
                 (platform, query))
    conn.commit()


def add_query(conn: sqlite3.Connection, platform: str, query: str, source: str,
              priority: float = 1.0, category: Optional[str] = None,
              family: Optional[str] = None, active: bool = True,
              cfg: Optional[dict] = None) -> bool:
    query = query.strip()
    if not query or query_policy_reason(query, cfg):
        return False
    cur = conn.execute(
        """INSERT OR IGNORE INTO queries
           (platform, query, source, priority, category, added_at, family, active)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (platform, query, source, priority, category, time.time(),
         family or infer_query_family(source, category), int(active)),
    )
    conn.commit()
    return cur.rowcount > 0


def record_query_proposal(
    conn: sqlite3.Connection, *, platform: str, query: str, source: str,
    decision, inserted: bool, parent_video_id: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    """Append one automatic-query decision for later manual audit."""
    conn.execute(
        """INSERT INTO query_proposals
           (proposed_at, platform, query, generation_source, parent_video_id,
            category, allowed, reason, policy_version, matched_text, inserted)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (time.time(), platform, query, source, parent_video_id, category,
         int(decision.allowed), decision.reason, decision.policy_version,
         decision.matched_text, int(inserted)),
    )
    conn.commit()


def record_snowball_proposal(
    conn: sqlite3.Connection, *, parent_video_id: str, platform: str,
    related_query: str, category: Optional[str], decision, inserted: bool,
) -> None:
    """Append one parent-gate decision for audit and transfer evaluation."""
    conn.execute(
        """INSERT INTO snowball_proposals
           (proposed_at, parent_video_id, platform, related_query, category,
            allowed, reason, policy_version, matched_text, evidence_json, inserted)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (time.time(), parent_video_id, platform, related_query, category,
         int(decision.allowed), decision.reason, decision.policy_version,
         decision.matched_text, json.dumps(decision.evidence or {}, sort_keys=True),
         int(inserted)),
    )
    conn.commit()


def _weighted_stale_pick(rows: list[sqlite3.Row], key: str,
                         weights: dict[str, float], now: float) -> Optional[str]:
    pick, best = None, -1.0
    for row in rows:
        value = str(row[key])
        weight = float(weights.get(value, 1.0))
        if weight <= 0:
            continue
        staleness = max(0.0, now - (row["recent"] or 0)) + 1.0
        score = weight * staleness + random.random()
        if score > best:
            pick, best = value, score
    return pick


def record_query_run(
    conn: sqlite3.Connection,
    platform: str,
    query: str,
    *,
    started_at: float,
    candidates_returned: int,
    new_candidates: int,
    enqueued_candidates: int,
    skipped_candidates: int,
    cursor_before: Optional[str],
    cursor_after: Optional[str],
    run_status: str = "complete",
    cfg: Optional[dict] = None,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """Persist one retrieval pass and impose a reversible zero-yield cooldown."""
    cfg = cfg or load_config()
    now = time.time() if now is None else now
    row = conn.execute(
        "SELECT family, zero_new_streak FROM queries WHERE platform=? AND query=?",
        (platform, query),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown query: {platform}:{query}")
    old_streak = int(row["zero_new_streak"] or 0)
    cooldown_cfg = ((cfg.get("scheduler", {}) or {}).get("zero_new_cooldown", {}) or {})
    enabled = bool(cooldown_cfg.get("enabled", True))
    cooldown_until = None
    if new_candidates > 0:
        streak = 0
        last_new_at = now
    elif run_status != "complete" or candidates_returned == 0:
        # A source outage/empty response is not evidence that the query text is
        # exhausted. Apply a short request-throttling cooldown without increasing
        # its saturation streak.
        streak = old_streak
        last_new_at = None
        if enabled:
            cooldown_until = now + float(cooldown_cfg.get("source_failure_seconds", 900))
    else:
        streak = old_streak + 1
        last_new_at = None
        if enabled:
            cursor_progressed = bool(cursor_after and cursor_after != cursor_before)
            base_key = "pagination_base_seconds" if cursor_progressed else "base_seconds"
            base = float(cooldown_cfg.get(base_key, 21600 if not cursor_progressed else 1800))
            maximum = float(cooldown_cfg.get("max_seconds", 259200))
            cooldown_until = now + min(maximum, base * (2 ** max(0, streak - 1)))

    conn.execute(
        """UPDATE queries SET last_used=?, zero_new_streak=?, cooldown_until=?,
                  last_new_at=COALESCE(?, last_new_at)
           WHERE platform=? AND query=?""",
        (now, streak, cooldown_until, last_new_at, platform, query),
    )
    already_seen = max(0, int(candidates_returned) - int(new_candidates))
    conn.execute(
        """INSERT INTO query_runs
           (platform, query, family, started_at, finished_at, run_status,
            cursor_before, cursor_after, candidates_returned, new_candidates,
            enqueued_candidates, skipped_candidates, already_seen,
            zero_new_streak, cooldown_until)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (platform, query, row["family"], started_at, now, run_status,
         cursor_before, cursor_after, int(candidates_returned), int(new_candidates),
         int(enqueued_candidates), int(skipped_candidates), already_seen,
         streak, cooldown_until),
    )
    conn.commit()
    return {"zero_new_streak": streak, "cooldown_until": cooldown_until,
            "already_seen": already_seen}


def next_query(conn: sqlite3.Connection, cfg: Optional[dict] = None) -> Optional[dict]:
    """Pick the next (platform, query).

    Platform is chosen by WEIGHTED STALENESS: each active platform scores
    weight * (seconds since it was last used). Staleness keeps it round-robin-ish
    (a just-used platform scores ~0, so it is not re-picked immediately), while a
    higher `scheduler.platform_weights` entry (e.g. reddit: 2.0) wins
    proportionally more picks -- reddit yields fewer videos per pick than
    dailymotion/rumble, so weighting its pick-rate up lifts its throughput. Within
    the chosen platform: highest priority, then LRU. Backward-compatible: with no
    cfg/weights every platform is weight 1.0 (plain staleness round-robin)."""
    cfg = cfg or {}
    now = time.time()
    ready = "active=1 AND COALESCE(policy_excluded,0)=0 AND " \
            "(cooldown_until IS NULL OR cooldown_until<=?)"
    family_weights = ((cfg.get("scheduler", {}) or {}).get("family_weights", {}) or {})
    if family_weights:
        family_rows = conn.execute(
            f"""WITH ready_families AS (
                   SELECT DISTINCT family FROM queries WHERE {ready}
                )
                SELECT q.family, MAX(COALESCE(q.last_used,0)) recent
                FROM queries q JOIN ready_families r ON r.family=q.family
                WHERE q.active=1 AND COALESCE(q.policy_excluded,0)=0
                GROUP BY q.family""",
            (now,),
        ).fetchall()
        family = _weighted_stale_pick(family_rows, "family", family_weights, now)
        if family is None:
            return None
        family_clause = " AND family=?"
        family_args: tuple[Any, ...] = (family,)
    else:
        family_clause = ""
        family_args = ()
    rows = conn.execute(
        f"""WITH ready_platforms AS (
               SELECT DISTINCT platform FROM queries
               WHERE {ready}{family_clause}
            )
            SELECT q.platform, MAX(COALESCE(q.last_used,0)) recent
            FROM queries q JOIN ready_platforms r ON r.platform=q.platform
            WHERE q.active=1 AND COALESCE(q.policy_excluded,0)=0{family_clause}
            GROUP BY q.platform""",
        (now, *family_args, *family_args),
    ).fetchall()
    if not rows:
        return None
    weights = ((cfg.get("scheduler", {}) or {}).get("platform_weights", {}) or {})
    pick = _weighted_stale_pick(rows, "platform", weights, now)
    if pick is None:
        return None
    row = conn.execute(
        f"""SELECT platform, query, cursor, category, source, family FROM queries
           WHERE {ready} AND platform=?{family_clause}
           ORDER BY priority DESC, COALESCE(last_used, 0) ASC, RANDOM()
           LIMIT 1""",
        (now, pick, *family_args),
    ).fetchone()
    return dict(row) if row else None


def query_cooldown_wait(conn: sqlite3.Connection, cfg: Optional[dict] = None,
                        now: Optional[float] = None) -> Optional[float]:
    """Seconds until a policy-eligible active query becomes runnable."""
    cfg = cfg or {}
    now = time.time() if now is None else now
    rows = conn.execute(
        """SELECT platform, family, cooldown_until FROM queries
           WHERE active=1 AND COALESCE(policy_excluded,0)=0
             AND cooldown_until>?""",
        (now,),
    ).fetchall()
    platform_weights = ((cfg.get("scheduler", {}) or {}).get("platform_weights", {}) or {})
    family_weights = ((cfg.get("scheduler", {}) or {}).get("family_weights", {}) or {})
    waits = [
        float(r["cooldown_until"]) - now for r in rows
        if float(platform_weights.get(r["platform"], 1.0)) > 0
        and float(family_weights.get(r["family"], 1.0)) > 0
    ]
    return min(waits) if waits else None


def query_stats(conn: sqlite3.Connection, platform: str, query: str):
    return conn.execute(
        "SELECT * FROM queries WHERE platform=? AND query=?", (platform, query)
    ).fetchone()


# --- reporting --------------------------------------------------------------

def report(conn: sqlite3.Connection, top: int = 25) -> dict:
    totals = conn.execute(
        """SELECT COUNT(*) AS videos_seen,
                  SUM(status='done') AS videos_done,
                  SUM(is_hit) AS videos_with_hits,
                  COALESCE(SUM(n_reactions), 0) AS total_reactions
           FROM seen_videos"""
    ).fetchone()

    by_source = conn.execute(
        """SELECT source,
                  COUNT(*) AS seen,
                  SUM(status='done') AS done,
                  SUM(is_hit) AS hits
           FROM seen_videos GROUP BY source ORDER BY hits DESC"""
    ).fetchall()

    top_queries = conn.execute(
        """SELECT platform, query, source, videos_processed, videos_with_hits,
                  total_reaction_count,
                  CASE WHEN videos_processed > 0
                       THEN 1.0*videos_with_hits/videos_processed ELSE 0 END AS hit_rate
           FROM queries WHERE videos_processed > 0
           ORDER BY hit_rate DESC, total_reaction_count DESC LIMIT ?""",
        (top,),
    ).fetchall()

    phrase_freq = conn.execute(
        """SELECT phrase, tier, tag, COUNT(*) AS n
           FROM reactions GROUP BY phrase ORDER BY n DESC LIMIT ?""",
        (top,),
    ).fetchall()

    return {
        "totals": dict(totals),
        "by_source": [dict(r) for r in by_source],
        "top_queries": [dict(r) for r in top_queries],
        "phrase_frequencies": [dict(r) for r in phrase_freq],
    }


def print_report(cfg: Optional[dict] = None) -> None:
    conn = init_db(cfg)
    print(json.dumps(report(conn), indent=2, default=str))
    conn.close()


def seed_taxonomy(conn: sqlite3.Connection,
                  platforms: Iterable[str] = ("dailymotion", "rumble")) -> int:
    """Seed the norm-taxonomy search terms (config/norm_taxonomy.yaml).

    Every term is tagged with its norm-category key so each surfaced video --
    and every snowball child it spawns -- is traceable to a norm class. Seeded as
    free-text search on dailymotion + rumble. (NOT rdsearch: anchored per-sub
    reddit search of the freakout subs only re-finds what the firehose already
    has -> 0 new videos; reddit clips get their norm class post-hoc from the LLM
    label instead.) Idempotent (INSERT OR IGNORE)."""
    try:
        tax = load_yaml("config/norm_taxonomy.yaml")
    except FileNotFoundError:
        return 0
    added = 0
    for key, spec in (tax.get("categories") or {}).items():
        terms = (spec or {}).get("terms", []) or []
        for plat in platforms:
            # priority 2.0 puts the curated taxonomy above generator exploit
            # queries (1.5) so the ~190 terms lead the dailymotion rotation and
            # are cycled BREADTH-FIRST via the LRU tiebreak in next_query.
            added += seed_queries(conn, terms, platform=plat,
                                  source="taxonomy", category=key, priority=2.0)
    return added


def seed_instructional(conn: sqlite3.Connection) -> int:
    """Seed the instructional vein (config/instructional_terms.yaml): teach-a-norm
    video terms on youtube+dailymotion + a few discussion subreddits, all tagged
    category='instr_<key>' so process_video routes them to the InstructionalDetector
    and the separate data/instructional/ corpus. Idempotent."""
    try:
        tax = load_yaml("config/instructional_terms.yaml")
    except FileNotFoundError:
        return 0
    platforms = tax.get("platforms", ["youtube", "dailymotion"])
    added = 0
    for key, spec in (tax.get("categories") or {}).items():
        terms = (spec or {}).get("terms", []) or []
        for plat in platforms:
            added += seed_queries(conn, terms, platform=plat, source="instructional",
                                  category=key, priority=1.4)
    for key, sublist in (tax.get("subreddits") or {}).items():
        added += seed_queries(conn, sublist or [], platform="reddit",
                              source="instructional", category=key, priority=1.1)
    return added


def seed_null(conn: sqlite3.Connection) -> int:
    """Seed curated mundane-twin queries for detector-cleared negatives."""
    try:
        tax = load_yaml("config/null_terms.yaml")
    except FileNotFoundError:
        return 0
    platforms = tax.get("platforms", ["dailymotion"])
    added = 0
    for key, spec in (tax.get("categories") or {}).items():
        terms = (spec or {}).get("terms", []) or []
        for plat in platforms:
            added += seed_queries(conn, terms, platform=plat, source="null",
                                  category=key, priority=1.2)
    return added


def finalize_instructional(conn: sqlite3.Connection, uid: str, n_demos: int = 0) -> None:
    """A didactic norm video: done, routed to the instructional text/clip corpus,
    NOT counted as a candid video hit."""
    conn.execute(
        """UPDATE seen_videos
           SET status='done', n_reactions=?, is_hit=0, modality='instructional',
               processed_at=?
           WHERE video_id=?""",
        (n_demos, time.time(), uid),
    )
    conn.commit()


def finalize_null(conn: sqlite3.Connection, uid: str, n_clips: int = 0) -> None:
    """Finalize a detector-cleared mundane-twin video as a negative."""
    conn.execute(
        """UPDATE seen_videos
           SET status='done', n_reactions=?, is_hit=0, modality='null_verified',
               processed_at=?
           WHERE video_id=?""",
        (n_clips, time.time(), uid),
    )
    conn.commit()


def seed_from_config(conn: sqlite3.Connection) -> dict:
    """Seed the queue from config/search_terms.yaml `sources` section plus the
    norm-taxonomy term bank.

    Each enabled source contributes its queries/subreddits at the configured
    platform. Returns a per-platform count of newly-added entries.
    """
    terms = load_yaml("config/search_terms.yaml")
    sources = terms.get("sources", {})
    added = {}
    for platform, spec in sources.items():
        if not spec or not spec.get("enabled", False):
            continue
        items = spec.get("queries") or spec.get("subreddits") or []
        added[platform] = seed_queries(conn, items, platform=platform)
    added["taxonomy"] = seed_taxonomy(conn)
    added["instructional"] = seed_instructional(conn)
    added["null"] = seed_null(conn)
    return added


if __name__ == "__main__":
    cfg = load_config()
    ensure_dirs(cfg)
    conn = init_db(cfg)
    added = seed_from_config(conn)
    print(f"DB ready at {resolve_path(cfg['paths']['state_db'])}; seeded: {added}")
    print_report(cfg)
