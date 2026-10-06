#!/usr/bin/env python3
"""Backfill audio-recovered reactions from metadata.json into SQLite.

audio_recover historically appended recovered reaction windows to each hit's
metadata.json but never wrote them to the DB, so report()'s SUM(n_reactions)
and the `reactions` table undercounted them. This reconciles every witnessed
hit that has audio_event reactions:

  * delete + reinsert that video's audio_event rows in `reactions`
  * set seen_videos.n_reactions = len(metadata reactions) (metadata = truth)

Idempotent — re-running converges to the same state. Safe to run while the
crawl is live (WAL + busy_timeout). Run from the repo root via run.sh:

    ./run.sh python scripts/backfill_audio_reactions.py --dry-run
    ./run.sh python scripts/backfill_audio_reactions.py
"""
import json
import sqlite3
import sys

sys.path.insert(0, ".")
from src import state  # noqa: E402


def main(dry_run: bool = False) -> None:
    cfg = state.load_config()
    hits_dir = state.resolve_path(cfg["paths"]["hits"])
    db = state.resolve_path(cfg["paths"]["state_db"])
    conn = sqlite3.connect(db, timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")

    n_vid = n_audio = 0
    for meta_p in sorted(hits_dir.glob("*/metadata.json")):
        uid = meta_p.parent.name
        try:
            meta = json.loads(meta_p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        reactions = meta.get("reactions", []) or []
        audio = [r for r in reactions
                 if r.get("source") == "audio_event" or r.get("tag") == "audio_event"]
        if not audio:
            continue
        n_vid += 1
        n_audio += len(audio)
        if dry_run:
            continue
        conn.execute(
            "DELETE FROM reactions WHERE video_id=? AND tag='audio_event'", (uid,))
        conn.executemany(
            """INSERT INTO reactions
               (video_id, clip_idx, phrase, tier, tag, start_time, end_time,
                speaker_id, context)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(uid, r.get("clip_idx"), r.get("phrase"), r.get("tier"),
              r.get("tag", "audio_event"), r.get("start"), r.get("end"),
              r.get("speaker"), r.get("context")) for r in audio])
        conn.execute(
            "UPDATE seen_videos SET n_reactions=?, is_hit=1 WHERE video_id=?",
            (len(reactions), uid))
    if not dry_run:
        conn.commit()
    conn.close()
    print(f"{'DRY ' if dry_run else ''}backfill: {n_vid} videos / "
          f"{n_audio} audio reactions reconciled")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
