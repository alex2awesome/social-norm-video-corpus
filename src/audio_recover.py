"""Pass A2 of the audio-event channel: recover NON-VERBAL reactions.

The transcript pipeline can only clip reactions someone SAID. Pass A found
strong vocal audio events (screams/gasps/cries) in many hits that sit nowhere
near an existing clip -- candidate reactions the transcript channel was deaf
to. This pass:

  1. enumerates candidate windows: vocal_reaction events with peak >=
     audio_events.recover_min_peak that don't fall within merge_window_sec of
     any known reaction (and respect the min-pre-roll rule), merged the same
     way extract_clips merges nearby reactions;
  2. asks the offline vLLM judge (same engine pattern as batch_detect) whether
     each window is a genuine spontaneous reaction to a real event -- the
     guard against music, concert crowds, rollercoaster screams, game audio;
  3. cuts a normal [t0-pre, t1+post] clip into the EXISTING hit dir for every
     "yes", appends the reaction to metadata.json (tag=audio_event), and
     DELETES any matched-negative clip the new reaction contaminates (a
     "no_reaction" control containing a scream is poison for the task).

Resumable via data/audio_recover_journal.jsonl (one line per finished uid).

    # candidate census only (CPU, no GPU):
    ./run.sh python -u -m src.audio_recover --dry-run
    # full run (ai_usage env, wrapper-style):
    CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. <ai_usage python> -u -m src.audio_recover
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

from . import state
from .batch_transcribe import _find_raw
from .clip_extract import _cut_copy, _cut_reencode
from .llm_detect import LLMDetector

log = logging.getLogger("audio_recover")

SYS = """You audit candidate moments in candid video footage. An audio classifier \
found loud VOCAL events (screaming, shouting, crying, gasping) at the timestamps \
below, in videos already known to contain social-norm violations. For EACH \
candidate window, decide whether the vocal event is a genuine SPONTANEOUS human \
reaction to something really happening in front of the camera (a fight, a crash, \
a confrontation, shocking behavior...).

Answer "no" when the loud vocals are instead: music or singing; a concert/sports \
crowd cheering the performance or game itself; amusement-ride screams; scripted \
or acted media; video-game/stream audio; a host yelling for the camera; ambient \
baby crying; laughter at a joke. The surrounding transcript may be empty -- \
non-verbal reactions are exactly what we are looking for, so an empty transcript \
is NOT evidence against a reaction.

Reply with STRICT JSON only:
{"windows": [{"idx": <int>, "is_reaction": "yes"|"no", "reason": "<short>", \
"norm": "<violated norm, snake_case, or null>"}]}"""


# --------------------------------------------------------------- candidates
def _segments_near(transcript: dict, t0: float, t1: float, pad: float = 30.0) -> str:
    segs = transcript.get("segments") or []
    out = [s.get("text", "").strip() for s in segs
           if s.get("end", 0) >= t0 - pad and s.get("start", 0) <= t1 + pad]
    return " ".join(x for x in out if x).strip()


def _candidates(cfg: dict, ae: dict, meta: dict) -> list[dict]:
    """Vocal events outside every known reaction, merged into clip windows."""
    acfg = cfg.get("audio_events", {})
    min_peak = float(acfg.get("recover_min_peak", 0.20))
    merge = cfg["clip"].get("merge_window_sec", 22)
    min_pre = cfg["clip"].get("min_pre_roll_sec", 10)
    reacts = [r.get("start") for r in meta.get("reactions", [])
              if r.get("start") is not None]
    evs = [e for e in ae.get("events", [])
           if e["group"] == "vocal_reaction" and e["peak"] >= min_peak
           and e["t0"] >= min_pre
           and not any(e["t0"] - merge <= r <= e["t1"] + merge for r in reacts)]
    evs.sort(key=lambda e: e["t0"])
    wins: list[dict] = []
    for e in evs:
        if wins and e["t0"] - wins[-1]["t1"] <= merge:
            w = wins[-1]
            w["t1"] = max(w["t1"], e["t1"])
            w["peak"] = max(w["peak"], e["peak"])
            w["classes"].add(e["class"])
        else:
            wins.append({"t0": e["t0"], "t1": e["t1"], "peak": e["peak"],
                         "classes": {e["class"]}})
    cap = int(acfg.get("recover_max_per_video", 4))
    wins.sort(key=lambda w: -w["peak"])
    wins = wins[:cap]
    wins.sort(key=lambda w: w["t0"])
    return wins


def _build_prompt(title: str, wins: list[dict], transcript: dict) -> list[dict]:
    parts = [f"VIDEO TITLE: {title or '(none)'}", ""]
    for i, w in enumerate(wins):
        txt = _segments_near(transcript, w["t0"], w["t1"]) or "(no intelligible speech nearby)"
        parts.append(f"CANDIDATE {i}: t={w['t0']}-{w['t1']}s, "
                     f"audio={'/'.join(sorted(w['classes']))} (peak {w['peak']:.2f})")
        parts.append(f"  transcript +/-30s: {txt[:700]}")
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": "\n".join(parts)}]


# ------------------------------------------------------------------ routing
def _next_clip_idx(hit_dir: Path) -> int:
    mx = -1
    for p in hit_dir.glob("clip_*.mp4"):
        try:
            mx = max(mx, int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return mx + 1


def _decontaminate_negatives(cfg: dict, uid: str, new_wins: list[dict]) -> int:
    """Delete matched-negative clips that overlap a newly recovered reaction."""
    neg_dir = state.resolve_path(cfg["paths"].get("negatives", "data/negatives")) / uid
    meta_p = neg_dir / "metadata.json"
    if not meta_p.exists():
        return 0
    clip_cfg = cfg["clip"]
    pre = clip_cfg.get("pre_seconds", 20)
    post = clip_cfg.get("post_seconds", 2)
    guard = clip_cfg.get("negatives", {}).get("guard_sec", 20)
    zones = [(w["t0"] - pre - guard, w["t1"] + post + guard) for w in new_wins]
    try:
        meta = json.loads(meta_p.read_text())
    except json.JSONDecodeError:
        return 0
    kept, removed = [], 0
    for n in meta.get("negatives", []):
        a, b = n.get("window", (None, None))
        if a is not None and any(not (b <= z0 or a >= z1) for z0, z1 in zones):
            (neg_dir / f"clip_{n['neg_idx']}.mp4").unlink(missing_ok=True)
            removed += 1
        else:
            kept.append(n)
    if removed:
        meta["negatives"] = kept
        meta["n_negatives"] = len(kept)
        meta.setdefault("audio_recover_removed", 0)
        meta["audio_recover_removed"] += removed
        meta["reaction_times"] = sorted(set(meta.get("reaction_times", []) +
                                            [round(w["t0"], 3) for w in new_wins]))
        meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        log.info("removed %d contaminated negatives for %s", removed, uid)
    return removed


def _save_recovered(cfg: dict, uid: str, media: Path, meta: dict,
                    wins: list[dict], verdicts: dict) -> list[dict]:
    """Cut a clip per confirmed window; append reactions to the hit metadata.

    Returns the list of reaction dicts appended (caller persists them to the DB)."""
    clip_cfg = cfg["clip"]
    pre = clip_cfg.get("pre_seconds", 20)
    post = clip_cfg.get("post_seconds", 2)
    reencode = clip_cfg.get("reencode_fallback", True)
    hit_dir = state.resolve_path(cfg["paths"]["hits"]) / uid
    if not hit_dir.is_dir():
        return 0
    saved_wins, saved_matches = [], []
    idx = _next_clip_idx(hit_dir)
    for i, w in enumerate(wins):
        v = verdicts.get(i) or {}
        if v.get("is_reaction") != "yes":
            continue
        ss = max(0.0, w["t0"] - pre)
        to = w["t1"] + post
        dst = hit_dir / f"clip_{idx}.mp4"
        ok = _cut_copy(media, dst, ss, to)
        if not ok and reencode:
            ok = _cut_reencode(media, dst, ss, to)
        if not ok:
            log.warning("could not cut recovered clip %d for %s", idx, uid)
            continue
        m = {
            "phrase": f"[audio] {'/'.join(sorted(w['classes']))}",
            "matched_text": None, "tier": None, "tag": "audio_event",
            "norm": v.get("norm"), "start": w["t0"], "end": w["t1"],
            "speaker": None, "context": (v.get("reason") or "")[:300],
            "source": "audio_event", "audio_peak": w["peak"],
            "clip_idx": idx, "clip_window": [round(ss, 3), round(to, 3)],
        }
        (hit_dir / f"reaction_{idx}.txt").write_text(
            "phrase:    {phrase}\nmatched:   (non-verbal audio event)\n"
            "tag:       audio_event\nreaction:  {t0:.3f}-{t1:.3f}s\n"
            "clip:      {ss:.3f}-{to:.3f}s\npeak:      {peak:.3f}\n"
            "llm:       {reason}\n".format(
                phrase=m["phrase"], t0=w["t0"], t1=w["t1"], ss=ss, to=to,
                peak=w["peak"], reason=v.get("reason", "")))
        meta.setdefault("reactions", []).append(m)
        saved_wins.append(w)
        saved_matches.append(m)
        idx += 1
    n_new = len(saved_matches)
    if n_new:
        meta["n_clips"] = meta.get("n_clips", 0) + n_new
        meta["n_reactions"] = meta.get("n_reactions", 0) + n_new
        meta["audio_recovered"] = meta.get("audio_recovered", 0) + n_new
        (hit_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False))
        _decontaminate_negatives(cfg, uid, saved_wins)
    return saved_matches


# --------------------------------------------------------------------- main
def run(dry_run: bool = False, limit: int = 0) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cfg = state.load_config()
    bcfg = cfg.get("batch", {})
    ae_dir = state.resolve_path(cfg.get("audio_events", {}).get("out_dir",
                                                                "data/audio_events"))
    hits_dir = state.resolve_path(cfg["paths"]["hits"])
    ts_dir = state.resolve_path(cfg["paths"]["transcripts"])
    journal_p = state.resolve_path("data/audio_recover_journal.jsonl")
    conn = sqlite3.connect(state.resolve_path(cfg["paths"]["state_db"]), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    titles = dict(conn.execute(
        "SELECT video_id, title FROM seen_videos WHERE title IS NOT NULL"))
    conn.close()
    done = set()
    if journal_p.exists():
        for line in journal_p.read_text().splitlines():
            try:
                done.add(json.loads(line)["uid"])
            except (json.JSONDecodeError, KeyError):
                pass

    # ---- enumerate candidates (CPU) --------------------------------------
    jobs = []          # (uid, media, meta, wins, messages)
    stats = {"scanned": 0, "done_before": 0, "no_candidates": 0,
             "raw_missing": 0, "candidates": 0}
    for ae_p in sorted(ae_dir.glob("*.json")):
        uid = ae_p.stem
        stats["scanned"] += 1
        if uid in done:
            stats["done_before"] += 1
            continue
        try:
            ae = json.loads(ae_p.read_text())
        except json.JSONDecodeError:
            continue
        if ae.get("error") or ae.get("source") != "raw" or not ae.get("events"):
            stats["no_candidates"] += 1
            continue
        meta_p = hits_dir / uid / "metadata.json"
        if not meta_p.exists():
            stats["no_candidates"] += 1
            continue
        meta = json.loads(meta_p.read_text())
        wins = _candidates(cfg, ae, meta)
        if not wins:
            stats["no_candidates"] += 1
            continue
        media = _find_raw(cfg, uid)
        if media is None:
            stats["raw_missing"] += 1
            continue
        transcript = {}
        ts_p = ts_dir / f"{uid}.json"
        if ts_p.exists():
            try:
                transcript = json.loads(ts_p.read_text())
            except json.JSONDecodeError:
                pass
        jobs.append((uid, media, meta, wins,
                     _build_prompt(titles.get(uid, ""), wins, transcript)))
        stats["candidates"] += len(wins)
        if limit and len(jobs) >= limit:
            break
    log.info("candidate census: %s -> %d videos / %d windows to judge",
             stats, len(jobs), stats["candidates"])
    if dry_run or not jobs:
        if dry_run and jobs:
            from collections import Counter
            peaks = sorted((w["peak"] for j in jobs for w in j[3]), reverse=True)
            cls = Counter(c for j in jobs for w in j[3] for c in w["classes"])
            log.info("window peaks: max=%.2f p50=%.2f; classes: %s",
                     peaks[0], peaks[len(peaks) // 2], dict(cls.most_common(8)))
        return

    # ---- offline vLLM judge (same guard pattern as batch_detect) ---------
    import torch
    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gb, total_gb = free_b / 2**30, total_b / 2**30
    need_gb = bcfg.get("engine_min_gb", 90)
    if free_gb < need_gb:
        log.warning("only %.0f GB free (< %d needed); exiting, journal untouched",
                    free_gb, need_gb)
        return
    util = min(bcfg.get("gpu_memory_utilization", 0.55), (free_gb - 6) / total_gb)
    log.info("GPU: %.0f/%.0f GB free -> gpu_memory_utilization=%.2f",
             free_gb, total_gb, util)
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=bcfg["model_path"],
              max_model_len=bcfg.get("max_model_len", 8192),
              gpu_memory_utilization=util)
    sp = SamplingParams(temperature=0, max_tokens=800)
    log.info("engine up in %.0fs; judging %d videos", time.time() - t0, len(jobs))
    outs = llm.chat([j[4] for j in jobs], sp)

    n_videos, n_clips, n_err = 0, 0, 0
    conn = sqlite3.connect(state.resolve_path(cfg["paths"]["state_db"]), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    with journal_p.open("a") as jf:
        for (uid, media, meta, wins, _msgs), out in zip(jobs, outs):
            parsed = LLMDetector._parse(out.outputs[0].text)
            if not isinstance(parsed, dict) or "windows" not in parsed:
                n_err += 1   # not journaled -> retried on a rerun
                continue
            verdicts = {v.get("idx"): v for v in parsed.get("windows", [])
                        if isinstance(v, dict)}
            recovered = _save_recovered(cfg, uid, media, meta, wins, verdicts)
            # persist recovered reactions to SQLite so report()/DB exports match
            # metadata.json (audio_recover historically wrote only the JSON)
            if recovered:
                state.add_recovered_reactions(conn, uid, recovered)
            n_videos += 1
            n_clips += len(recovered)
            jf.write(json.dumps({"uid": uid, "n_windows": len(wins),
                                 "n_recovered": len(recovered)}) + "\n")
    conn.close()
    log.info("Pass A2 done in %.0f min: %d videos judged, %d clips recovered, "
             "%d parse errors (will retry)",
             (time.time() - t0) / 60, n_videos, n_clips, n_err)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recover non-verbal reactions from audio events")
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate candidates only (CPU, no GPU)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
