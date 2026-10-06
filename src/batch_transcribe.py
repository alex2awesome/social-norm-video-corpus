"""Batch STAGE 1: transcribe the pending queue, then release the GPU.

Part of the GPU-free-crawl design (llm.deferred): the crawl only enumerates and
downloads (status='pending_transcribe'); this job loads WhisperX once, chews
through the backlog, marks rows 'pending_detect' for stage 2 (batch_detect),
and exits -- process exit guarantees the GPU is fully released.

Runs from the norm-scraper env (WhisperX/torch). The wrapper
(batch_pipeline.sh) picks a free GPU and exports CUDA_VISIBLE_DEVICES before
launching, so transcribe.device_index is forced to 0 here.

    ./run.sh python -u -m src.batch_transcribe [--limit N]
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

from . import scrape, state

log = logging.getLogger("batch_ts")

RAW_EXTS = (".mp4", ".mkv", ".webm", ".m4a", ".mov")


def _find_raw(cfg, uid: str) -> Path | None:
    raw_dir = state.resolve_path(cfg["paths"]["raw_video"])
    for p in raw_dir.glob(f"{uid}.*"):
        if p.suffix.lower() in RAW_EXTS:
            return p
    return None


def run(limit: int = 0) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cfg = state.load_config()
    # the wrapper already pinned CUDA_VISIBLE_DEVICES to the chosen GPU
    cfg["transcribe"]["device_index"] = 0

    conn = sqlite3.connect(state.resolve_path(cfg["paths"]["state_db"]), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")

    rows = conn.execute(
        "SELECT video_id FROM seen_videos WHERE status='pending_transcribe' "
        "ORDER BY enumerated_at").fetchall()
    if not rows:
        log.info("nothing pending; exiting without touching the GPU")
        return
    if limit:
        rows = rows[:limit]
    log.info("batch transcribe: %d pending", len(rows))

    from .transcribe import Transcriber   # GPU load deferred to here
    transcriber = Transcriber(cfg)

    t0, n_ok, n_err = time.time(), 0, 0
    for r in rows:
        uid = r["video_id"]
        media = _find_raw(cfg, uid)
        if media is None:
            log.warning("raw missing for %s; marking error", uid)
            state.set_status(conn, uid, "error", error="raw missing at batch transcribe")
            n_err += 1
            continue
        try:
            transcriber.transcribe(media, uid)
        except Exception as e:
            msg = str(e)
            if "CUDA" in msg or "out of memory" in msg.lower():
                # TRANSIENT: another job grabbed the GPU mid-batch. Keep the row
                # pending (raw intact) and abort -- the wrapper/cron retries when
                # a GPU is genuinely free.
                log.warning("CUDA/OOM during %s -- GPU contended; aborting stage "
                            "(row stays pending): %s", uid, msg[:160])
                break
            log.exception("transcription failed for %s", uid)
            state.set_status(conn, uid, "error", error=msg[:500])
            scrape.purge_raw(uid, cfg)
            n_err += 1
            continue
        state.set_status(conn, uid, "pending_detect")
        n_ok += 1
        if n_ok % 50 == 0:
            log.info("transcribed %d/%d (%.1fs/video)", n_ok, len(rows),
                     (time.time() - t0) / max(1, n_ok))
    log.info("stage 1 done: %d transcribed, %d errors, %.0f min",
             n_ok, n_err, (time.time() - t0) / 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch transcription stage")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
