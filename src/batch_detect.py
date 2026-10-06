"""Batch STAGE 2: judge the transcribed queue with an OFFLINE vLLM engine.

Part of the GPU-free-crawl design (llm.deferred): loads Llama-3.3-70B-FP8 via
vllm.LLM, runs every pending transcript through the SAME detectors and routing
as the live loop (LLMDetector/InstructionalDetector prepare()+finish(),
search_loop.resolve_llm_result + finish_detection), then exits -- process exit
releases the GPU completely. Continuous batching makes this far faster than the
old serial HTTP server (hundreds of transcripts in minutes).

LLM errors leave rows at status='pending_detect' (retried next batch).
Hit/reaction query stats are applied here, hours after the crawl counted the
video as processed -- the scheduler tolerates the lag.

Runs from the ai_usage env (the only env whose vllm imports); the wrapper sets
CUDA_VISIBLE_DEVICES, PYTHONPATH (project root) and PATH (norm-scraper bin, for
ffmpeg) before launching:

    batch_pipeline.sh            # normal path (cron)
    .../ai_usage/bin/python -u -m src.batch_detect [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

from . import detect_reactions, state
from .batch_transcribe import _find_raw
from .llm_detect import LLMDetector, InstructionalDetector
from .search_loop import finish_detection, resolve_llm_result

log = logging.getLogger("batch_llm")


def _load_transcript(cfg, uid: str):
    p = state.resolve_path(cfg["paths"]["transcripts"]) / f"{uid}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _clamp_messages(tok, messages, budget: int) -> bool:
    """Token-budget the transcript inside the user message so the rendered chat
    prompt fits `budget` tokens. Returns True if it had to truncate.

    The detector truncates the transcript by CHARACTERS (llm.max_chars), which is
    safe for English (~0.25 tok/char) but NOT for dense scripts: CJK/Cyrillic/
    Arabic tokenize at ~1 tok/char, so a single long non-English transcript can
    exceed max_model_len. vLLM validates the whole batch at once, so one over-long
    prompt kills the engine core and the entire batch produces nothing. This guard
    makes the budget language-agnostic. The transcript sits between the first and
    last triple-quote in the user content (see LLMDetector._user_prompt)."""
    def ntok():
        return len(tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True))

    if ntok() <= budget:
        return False
    user = messages[-1]["content"]
    a = user.find('"""')
    b = user.rfind('"""')
    if a < 0 or b <= a + 3:
        return False  # no transcript region found (shouldn't happen) -> leave as-is
    head, body, tail = user[:a + 3], user[a + 3:b], user[b:]
    lo, hi = 0, len(body)          # binary-search the largest body prefix that fits
    while hi - lo > 64:
        mid = (lo + hi) // 2
        messages[-1] = {"role": "user", "content": head + body[:mid] + tail}
        if ntok() <= budget:
            lo = mid
        else:
            hi = mid
    messages[-1] = {"role": "user", "content": head + body[:lo] + tail}
    return True


def _cand_from_row(row) -> dict:
    """Reconstruct the candidate dict finish_detection expects from seen_videos."""
    uid = row["video_id"]
    return {
        "uid": uid,
        "source": row["source"],
        "url": row["url"],
        "title": row["title"],
        "duration": row["duration"],
        # uid format is "<platform>__<native_id>" (snowball needs native_id)
        "native_id": uid.split("__", 1)[-1],
    }


def run(limit: int = 0) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cfg = state.load_config()
    bcfg = cfg.get("batch", {})
    conn = sqlite3.connect(state.resolve_path(cfg["paths"]["state_db"]), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")

    rows = conn.execute(
        "SELECT * FROM seen_videos WHERE status='pending_detect' "
        "ORDER BY enumerated_at").fetchall()
    if limit:
        rows = rows[:limit]
    if not rows:
        log.info("nothing pending; exiting without touching the GPU")
        return
    log.info("batch detect: %d pending transcripts", len(rows))

    det = LLMDetector(cfg)
    idet = InstructionalDetector(cfg)
    kw = detect_reactions.ReactionDetector(cfg)

    # ---- build round-1 jobs (instr_* rows get the instructional prompt) ------
    jobs = []          # (row, transcript, kind, messages, aux)
    counts: dict = {}

    def bump(k):
        counts[k] = counts.get(k, 0) + 1

    for row in rows:
        transcript = _load_transcript(cfg, row["video_id"])
        if transcript is None:
            state.set_status(conn, row["video_id"], "error", error="transcript missing at batch detect")
            bump("no_transcript")
            continue
        instr = row["category"] and str(row["category"]).startswith("instr")
        d = idet if instr else det
        messages, aux = d.prepare(transcript, row["title"])
        if messages is None:   # too short for a call -> resolve immediately as miss
            _route(conn, cfg, kw, row, transcript, aux if not instr else
                   {"is_target": "no", "hard_negative_type": "other", "matches": []},
                   counts)
            continue
        jobs.append((row, transcript, "instr" if instr else "candid", messages, aux))

    if not jobs:
        log.info("no LLM calls needed: %s", counts)
        return

    # ---- load the offline engine (this is the GPU-heavy moment) --------------
    # The wrapper gated on free memory, but another job may have grabbed the GPU
    # since: re-measure NOW, size the engine to what is actually free, and exit
    # gracefully (rows stay pending) if it no longer fits.
    import torch
    free_b, total_b = torch.cuda.mem_get_info(0)   # CUDA_VISIBLE_DEVICES pins us
    free_gb, total_gb = free_b / 2**30, total_b / 2**30
    need_gb = bcfg.get("engine_min_gb", 90)
    if free_gb < need_gb:
        log.warning("only %.0f GB free on the assigned GPU (< %d needed) -- "
                    "grabbed since the wrapper checked; exiting, rows stay pending",
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
    out_tokens = cfg["llm"].get("max_tokens", 1600)
    sp = SamplingParams(temperature=0, max_tokens=out_tokens)
    log.info("engine up in %.0fs; round 1: %d prompts", time.time() - t0, len(jobs))

    # Hard token-budget guard: a single over-length prompt (dense non-English
    # transcripts tokenize far past the char-based max_chars) would otherwise
    # crash the whole offline engine. Clamp every prompt to fit max_model_len.
    tok = llm.get_tokenizer()
    budget = bcfg.get("max_model_len", 8192) - out_tokens - 16
    n_clamped = sum(_clamp_messages(tok, j[3], budget) for j in jobs)
    if n_clamped:
        log.info("clamped %d over-long prompt(s) to <=%d input tokens", n_clamped, budget)

    outs = llm.chat([j[3] for j in jobs], sp)
    round2 = []
    for (row, transcript, kind, _msgs, aux), out in zip(jobs, outs):
        parsed = LLMDetector._parse(out.outputs[0].text)
        if kind == "instr":
            ires = idet.finish(parsed, aux)
            if ires.get("is_instructional") == "yes" and ires.get("demos"):
                _route_instructional(conn, cfg, row, ires, counts)
                continue
            # not actually instructional -> candid detection in round 2
            messages, aux2 = det.prepare(transcript, row["title"])
            if messages is None:
                _route(conn, cfg, kw, row, transcript, aux2, counts)
            else:
                round2.append((row, transcript, messages, aux2))
            continue
        res = det.finish(parsed, aux)
        _route(conn, cfg, kw, row, transcript, res, counts)

    if round2:
        log.info("round 2 (instr fallbacks -> candid): %d prompts", len(round2))
        for j in round2:
            _clamp_messages(tok, j[2], budget)
        outs2 = llm.chat([j[2] for j in round2], sp)
        for (row, transcript, _msgs, aux), out in zip(round2, outs2):
            res = det.finish(LLMDetector._parse(out.outputs[0].text), aux)
            _route(conn, cfg, kw, row, transcript, res, counts)

    log.info("stage 2 done in %.0f min: %s",
             (time.time() - t0) / 60, json.dumps(counts, sort_keys=True))


def _route_instructional(conn, cfg, row, ires, counts) -> None:
    from . import clip_extract
    uid = row["video_id"]
    media = _find_raw(cfg, uid)
    cand = _cand_from_row(row)
    prov = {"platform": row["source"], "found_by_query": row["query"],
            "query_source": row["query_source"], "category": row["category"]}
    n = 0
    if media is not None:
        n = clip_extract.save_instructional(media, uid, ires, cfg, cand,
                                            row["category"], prov=prov)
    state.finalize_instructional(conn, uid, n)
    state.update_query_stats(conn, row["platform"], row["query"], 0,
                             1 if n > 0 else 0, n)
    counts["instructional"] = counts.get("instructional", 0) + 1


def _route(conn, cfg, kw, row, transcript, res, counts) -> None:
    """Resolve one candid result and run the shared routing. LLM error -> leave
    the row pending for the next batch."""
    from . import scrape
    uid = row["video_id"]
    if res.get("is_target") == "error":
        counts["llm_error"] = counts.get("llm_error", 0) + 1
        return   # stays pending_detect
    matches, verdict, modality, agent, expand, scene = resolve_llm_result(
        cfg, kw, transcript, res)
    cand = _cand_from_row(row)
    prov = {"platform": row["source"], "found_by_query": row["query"],
            "query_source": row["query_source"], "category": row["category"]}
    media = _find_raw(cfg, uid)
    if media is None:
        # raw vanished between download and batch -> can't clip; count as a miss
        log.warning("raw missing for %s at batch detect; finalizing as miss", uid)
        state.finalize_video(conn, uid, 0)
        counts["raw_missing"] = counts.get("raw_missing", 0) + 1
        return
    n = finish_detection(conn, cfg, uid, cand, media, row["query"],
                         row["category"], row["query_source"], prov,
                         matches, verdict, modality, agent, expand, scene)
    state.update_query_stats(conn, row["platform"], row["query"], 0,
                             1 if n > 0 else 0, n)
    key = ("hit" if n > 0 else
           "commentary" if modality == "commentary" else
           "drop" if str(verdict).startswith("drop") else "miss")
    counts[key] = counts.get(key, 0) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch LLM detection stage (offline vLLM)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
