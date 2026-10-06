"""Pass A of the audio-event channel: score every existing HIT's audio.

For each hit directory, run PANNs SED (src.audio_events) over its raw video
(falling back to the saved clips if the raw is gone), write the full timeline
to data/audio_events/<uid>.json, and stamp the summary (scream_peak /
commotion_peak / interest) into the hit's metadata.json so the corpus can be
reranked without re-reading the timelines.

RESUMABLE: a uid with an existing output json is skipped, so the pass can be
killed and relaunched freely. Per-video failures are recorded in the json
({"error": ...}) so they don't retry forever.

    CUDA_VISIBLE_DEVICES=<gpu> ./run.sh python -u -m src.batch_audio_events [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from . import state
from .batch_transcribe import _find_raw

log = logging.getLogger("batch_audio")


def _stamp_metadata(hit_dir, summary) -> None:
    meta_p = hit_dir / "metadata.json"
    if not meta_p.exists():
        return
    try:
        meta = json.loads(meta_p.read_text())
    except json.JSONDecodeError:
        return
    meta["audio_events"] = summary
    meta_p.write_text(json.dumps(meta, indent=2))


def run(limit: int = 0) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cfg = state.load_config()
    out_dir = state.resolve_path(cfg.get("audio_events", {}).get("out_dir",
                                                                 "data/audio_events"))
    out_dir.mkdir(parents=True, exist_ok=True)
    hits_dir = state.resolve_path(cfg["paths"]["hits"])

    uids = sorted(p.name for p in hits_dir.iterdir() if p.is_dir())
    todo = [u for u in uids if not (out_dir / f"{u}.json").exists()]
    log.info("audio-event Pass A: %d hits, %d already scored, %d to do",
             len(uids), len(uids) - len(todo), len(todo))
    if limit:
        todo = todo[:limit]
    if not todo:
        return

    from .audio_events import AudioEventDetector   # GPU load deferred to here
    det = AudioEventDetector(cfg)

    t0, n_ok, n_clip, n_err = time.time(), 0, 0, 0
    for uid in todo:
        hit_dir = hits_dir / uid
        out_p = out_dir / f"{uid}.json"
        try:
            media = _find_raw(cfg, uid)
            if media is not None:
                result = det.analyze(media)
                source = "raw"
            else:
                # raw purged -> score the saved clips (timeline is per-clip,
                # but the peaks still rank the hit)
                result, source = None, "clips"
                clips = {}
                for cp in sorted(hit_dir.glob("clip_*.mp4")):
                    r = det.analyze(cp)
                    if r:
                        clips[cp.name] = r
                if clips:
                    best = {k: round(max(c["summary"][k] for c in clips.values()), 4)
                            for k in ("scream_peak", "commotion_peak", "interest")}
                    best["n_events"] = sum(c["summary"]["n_events"]
                                           for c in clips.values())
                    result = {"clips": clips, "summary": best}
                n_clip += 1
            if result is None:
                out_p.write_text(json.dumps({"error": "no decodable audio"}))
                n_err += 1
                continue
            result["source"] = source
            out_p.write_text(json.dumps(result, indent=2))
            _stamp_metadata(hit_dir, result["summary"])
            n_ok += 1
        except Exception as e:
            msg = str(e)
            if "CUDA" in msg or "out of memory" in msg.lower():
                # TRANSIENT: another job grabbed the GPU mid-pass. Don't write
                # an error json (uid stays todo) -- abort, relaunch later.
                log.warning("CUDA/OOM at %s -- GPU contended; aborting pass: %s",
                            uid, msg[:160])
                break
            log.exception("audio-event scoring failed for %s", uid)
            out_p.write_text(json.dumps({"error": msg[:500]}))
            n_err += 1
        done = n_ok + n_err
        if done % 25 == 0:
            log.info("scored %d/%d (%.1fs/video, %d from clips, %d errors)",
                     done, len(todo), (time.time() - t0) / max(1, done),
                     n_clip, n_err)
    log.info("Pass A done: %d scored (%d from clips), %d errors, %.0f min",
             n_ok, n_clip, n_err, (time.time() - t0) / 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audio-event Pass A over existing hits")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
