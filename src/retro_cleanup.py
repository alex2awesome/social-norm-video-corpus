"""Retroactive cleanup + backfill of the WITNESSED corpus (Phase 1 of PLAN.md).

Re-runs the CURRENT detector (scripted_media drop, narration+solo->commentary,
severity/reaction_strength, LLM-only matches) over every existing hit dir, using
the transcripts already on disk -- no download, no GPU transcription, just LLM
calls to the local vLLM endpoint shared with the live crawl (throttled).

Per video, in order:
  1. DEDUP   same normalized title + duration±2s as an earlier hit -> quarantine.
  2. TITLE   _TITLE_SKIP junk (full film / gameplay / wrestling / compilation)
             -> quarantine without an LLM call.
  3. DETECT  re-run LLMDetector on the saved transcript:
       drop:*      -> quarantine (reason = hard_negative_type)
       commentary  -> write a discussion-corpus record, quarantine the hit dir
       witnessed   -> if the raw video is still on disk: move the old dir to
                      quarantine/pre_recut and RE-CUT clips under the new rules
                      (LLM-only matches, merge<22s, min pre-roll), refresh the
                      negatives pool; else annotate-only (stamp scene/severity
                      into the existing metadata.json).

NOTHING is hard-deleted: every removed dir moves to data/quarantine/<reason>/.
Resumable via data/retro_journal.jsonl (LLM errors are retried on rerun).

Run (on sk3, behind the live crawl):
    ./run.sh python -u -m src.retro_cleanup --limit 5      # smoke test
    nohup nice -n 15 ./run.sh python -u -m src.retro_cleanup >> logs/retro.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
import time
from pathlib import Path

from . import clip_extract, state
from .llm_detect import LLMDetector
from .search_loop import _TITLE_SKIP

log = logging.getLogger("retro")

RAW_EXTS = (".mp4", ".mkv", ".webm", ".m4a", ".mov")


def _journal_path(cfg) -> Path:
    return state.resolve_path("data/retro_journal.jsonl")


def _load_journal(path: Path) -> dict:
    """uid -> action for completed entries (llm_error entries are retried)."""
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("action") != "llm_error":
                done[rec["uid"]] = rec["action"]
    return done


def _log_journal(path: Path, uid: str, action: str, reason: str = None, **extra):
    rec = {"uid": uid, "action": action, "reason": reason, "ts": time.time(), **extra}
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _quarantine(cfg, uid: str, reason: str) -> None:
    """Move hits/<uid> (and its matched negatives) under data/quarantine/<reason>/.
    Same-filesystem rename: instant, nothing is lost."""
    qdir = state.resolve_path("data/quarantine") / reason
    qdir.mkdir(parents=True, exist_ok=True)
    hit = state.resolve_path(cfg["paths"]["hits"]) / uid
    if hit.exists():
        shutil.move(str(hit), str(qdir / uid))
    neg = state.resolve_path(cfg["paths"].get("negatives", "data/negatives")) / uid
    if neg.exists():
        (qdir / "_negatives").mkdir(exist_ok=True)
        shutil.move(str(neg), str(qdir / "_negatives" / uid))


def _find_raw(cfg, uid: str) -> Path | None:
    raw_dir = state.resolve_path(cfg["paths"]["raw_video"])
    for p in raw_dir.glob(f"{uid}.*"):
        if p.suffix.lower() in RAW_EXTS:
            return p
    return None


def _db_mark(conn, uid: str, modality: str, n_reactions: int = 0) -> None:
    conn.execute(
        """UPDATE seen_videos SET is_hit=?, n_reactions=?, modality=?, processed_at=?
           WHERE video_id=?""",
        (1 if n_reactions > 0 else 0, n_reactions, modality, time.time(), uid))
    conn.commit()


def _row(conn, uid: str):
    return conn.execute(
        "SELECT title, url, source, agent, duration, processed_at FROM seen_videos "
        "WHERE video_id=?", (uid,)).fetchone()


def _dedup_losers(conn, uids: set[str]) -> set[str]:
    """Among current witnessed hits, group by (normalized title, ~duration);
    every member except the EARLIEST-processed is a duplicate."""
    rows = conn.execute(
        """SELECT video_id, lower(trim(title)) t, duration d, processed_at p
           FROM seen_videos WHERE is_hit=1 AND modality='witnessed'
             AND title IS NOT NULL AND duration IS NOT NULL""").fetchall()
    rows = [r for r in rows if r["video_id"] in uids and r["t"]]
    groups: dict = {}
    for r in sorted(rows, key=lambda r: r["p"] or 0):
        # duration bucketed to 2s so ±2s re-uploads collide
        groups.setdefault((r["t"], round((r["d"] or 0) / 2)), []).append(r["video_id"])
    return {uid for g in groups.values() for uid in g[1:]}


def run(limit: int = 0, throttle: float = 1.0) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cfg = state.load_config()
    db_path = state.resolve_path(cfg["paths"]["state_db"])
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")

    detector = LLMDetector(cfg)
    hits_dir = state.resolve_path(cfg["paths"]["hits"])
    journal = _journal_path(cfg)
    done = _load_journal(journal)

    uids = sorted(d.name for d in hits_dir.iterdir() if d.is_dir())
    dupes = _dedup_losers(conn, set(uids))
    log.info("retro: %d hit dirs (%d already journaled, %d title+duration dupes)",
             len(uids), len(done), len(dupes))

    counts: dict = {}
    n_run = 0
    for uid in uids:
        if uid in done:
            continue
        if limit and n_run >= limit:
            break
        n_run += 1
        row = _row(conn, uid)
        title = (row["title"] if row else None) or ""

        # 1. content dedup (keep the earliest upload)
        if uid in dupes:
            _quarantine(cfg, uid, "duplicate")
            conn.execute("UPDATE seen_videos SET is_hit=0, n_reactions=0, "
                         "status='duplicate' WHERE video_id=?", (uid,))
            conn.commit()
            _log_journal(journal, uid, "duplicate", title=title[:80])
            counts["duplicate"] = counts.get("duplicate", 0) + 1
            continue

        # 2. junk-title sweep (compilations / full films / gameplay / wrestling)
        if title and _TITLE_SKIP.search(title):
            _quarantine(cfg, uid, "title_filter")
            _db_mark(conn, uid, "dropped")
            _log_journal(journal, uid, "title_filter", title=title[:80])
            counts["title_filter"] = counts.get("title_filter", 0) + 1
            continue

        # 3. re-detect from the saved transcript
        tpath = state.resolve_path(cfg["paths"]["transcripts"]) / f"{uid}.json"
        if not tpath.exists():
            _log_journal(journal, uid, "no_transcript")
            counts["no_transcript"] = counts.get("no_transcript", 0) + 1
            continue
        try:
            transcript = json.loads(tpath.read_text())
        except json.JSONDecodeError:
            _log_journal(journal, uid, "bad_transcript")
            counts["bad_transcript"] = counts.get("bad_transcript", 0) + 1
            continue

        res = detector.detect(transcript, title=title or None)
        time.sleep(throttle)  # stay polite to the endpoint the live crawl shares
        verdict = res.get("is_target")
        if verdict == "error":
            _log_journal(journal, uid, "llm_error")   # retried on next run
            counts["llm_error"] = counts.get("llm_error", 0) + 1
            time.sleep(10)
            continue

        # old provenance survives every route
        old_meta = {}
        mpath = hits_dir / uid / "metadata.json"
        if mpath.exists():
            try:
                old_meta = json.loads(mpath.read_text())
            except json.JSONDecodeError:
                pass
        prov = dict(old_meta.get("provenance") or {})
        prov.pop("modality", None)
        if res.get("scene"):
            prov["scene"] = res["scene"]
        agent = res.get("agent") or old_meta.get("agent") or "human"

        if verdict != "yes":
            reason = "drop_" + str(res.get("hard_negative_type") or "other")
            _quarantine(cfg, uid, reason)
            _db_mark(conn, uid, "dropped")
            _log_journal(journal, uid, "dropped", reason=reason, title=title[:80])
            counts[reason] = counts.get(reason, 0) + 1
            continue

        if res.get("modality") == "commentary":
            cand = {"url": row["url"] if row else None, "title": title,
                    "source": row["source"] if row else None}
            clip_extract.save_discussion(uid, res.get("matches", []), cfg, cand,
                                         agent=agent, prov=prov)
            state.finalize_discussion(conn, uid, agent=agent)
            _quarantine(cfg, uid, "commentary_rerouted")
            _log_journal(journal, uid, "commentary", title=title[:80])
            counts["commentary"] = counts.get("commentary", 0) + 1
            continue

        # witnessed -> re-cut under the new rules when the raw is still here
        raw = _find_raw(cfg, uid)
        matches = res.get("matches", [])
        if raw is not None and matches:
            _quarantine(cfg, uid, "pre_recut")  # old clips kept, never lost
            saved = clip_extract.extract_clips(raw, uid, matches, cfg,
                                               row["url"] if row else None,
                                               agent=agent, prov=prov)
            if saved:
                clip_extract.sample_negatives(raw, uid, saved, cfg,
                                              row["url"] if row else None, prov=prov)
                conn.execute("DELETE FROM reactions WHERE video_id=?", (uid,))
                conn.commit()
                state.record_reactions(conn, uid, saved)
                state.finalize_video(conn, uid, len(saved), modality="witnessed",
                                     agent=agent)
                _log_journal(journal, uid, "recut", n_reactions=len(saved))
                counts["recut"] = counts.get("recut", 0) + 1
            else:
                # every group truncated -> not usable as a hit
                _db_mark(conn, uid, "dropped")
                _log_journal(journal, uid, "dropped", reason="all_clips_truncated")
                counts["all_clips_truncated"] = counts.get("all_clips_truncated", 0) + 1
            continue

        # raw gone (or no locatable quotes): keep the old clips, BACKFILL the
        # annotation so the dir is at least filterable/rankable downstream
        if old_meta:
            old_meta["provenance"] = {**prov, "modality": "witnessed", "agent": agent}
            old_meta["retro_annotated"] = True
            mpath.write_text(json.dumps(old_meta, indent=2, ensure_ascii=False))
        _log_journal(journal, uid, "annotated",
                     reason="no_raw" if raw is None else "no_locatable_quotes")
        counts["annotated"] = counts.get("annotated", 0) + 1

    log.info("retro pass done (%d examined this run): %s",
             n_run, json.dumps(counts, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="Retroactive witnessed-corpus cleanup")
    ap.add_argument("--limit", type=int, default=0, help="stop after N videos (0 = all)")
    ap.add_argument("--throttle", type=float, default=1.0,
                    help="sleep between LLM calls (s); the live crawl shares the endpoint")
    args = ap.parse_args()
    run(limit=args.limit, throttle=args.throttle)


if __name__ == "__main__":
    main()
