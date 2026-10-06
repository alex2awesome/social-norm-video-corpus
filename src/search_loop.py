"""Long-running search loop.

For each query popped from the priority queue:
  enumerate candidates -> dedupe -> download -> transcribe (ALWAYS save) ->
  detect reactions -> if hits: clip + keep raw; else: purge raw -> update stats.

Every `loop.eval_every` processed videos, the query generator prunes losers and
adds exploit/explore queries. The process is resumable: state lives in the DB,
transcripts/clips on disk; a restart skips anything already `done`.

Run:
    python -m src.search_loop                 # run forever
    python -m src.search_loop --max-videos 5  # bounded (smoke test)
    python -m src.search_loop --report        # print state.db summary and exit
    python -m src.search_loop --no-download    # enumerate+stats only (no media/GPU)
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from . import clip_extract, detect_reactions, query_generator, scrape, sources, state

log = logging.getLogger("loop")

_STOP = False


def _handle_signal(signum, frame):
    global _STOP
    _STOP = True
    log.warning("signal %s received; finishing current video then exiting", signum)


def setup_logging(cfg: dict) -> None:
    log_dir = state.resolve_path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    # date is derived from the OS, not Python's clock-restricted helpers
    day = time.strftime("%Y-%m-%d")
    handlers = [
        logging.FileHandler(log_dir / f"run_{day}.log"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def _too_long(cand: dict, cfg: dict) -> bool:
    """Reject by enumeration metadata before spending any bandwidth."""
    dur = cand.get("duration")
    cap = cfg["search"].get("max_duration_sec", 1800)
    return dur is not None and dur > cap


# Title-level junk filter: scripted media (full movies / TV episodes), video-game
# content, pro wrestling, and multi-incident compilations all polluted the
# witnessed corpus ("Strangers in a Bed - Full Film" saved 3x as witnessed hits;
# GTA roleplay matched "laughing at homeless man"). Skipping by title costs zero
# bandwidth; the LLM scripted_media verdict is the backstop for unlabeled ones.
import re as _re
_TITLE_SKIP = _re.compile(
    r"\bfull\s+(movie|film|episode)\b|\bepisode\s+\d+\b|\bofficial\s+trailer\b"
    r"|\bgameplay\b|\bwalkthrough\b|\blet'?s\s+play\b|\bgta\s*[v5]?\b|\bfivem\b"
    r"|\bminecraft\b|\broblox\b|\bwwe\b|\baew\b|\bsmackdown\b|\bwrestl"
    r"|\bcompilation\b"
    # scripted-TV / series markers (QA 2026-06-14: named soaps/dramas, Dhar Mann
    # studio skits, and #N dashcam/road-rage *series* leaked into witnessed):
    r"|\bs\d{1,2}e\d{1,2}\b|\bep\.?\s*#?\s*\d+|\bdhar\s*mann\b"
    r"|(?:dash\s*cam|bad\s+driv|idiot\s+driv|road\s+rage|car\s+crashes?)[^\n]*#\s*\d+"
    # QA 2026-06-14b: dailymotion 'related' expansion leaked scripted comedy
    # skits and synthetic narration -- Vine compilations, pick-up-line bits, and
    # AI-narrated reddit-story channels (acted/synthetic, never witnessed):
    r"|\bvines\b|\bpick.?up\s+lines?\b|\breddit\s+(?:cheating|stor(?:y|ies))\b|#reddit\b"
    # QA 2026-06-14: named scripted shows that recur via dailymotion 'related'
    # and evade generic patterns (teleserye/soap titles). Add repeat offenders:
    r"|\bwalang\s+hanggang\s+paalam\b"
    # QA 2026-06-19/21: GoAnimate "grounded" animated kids videos recur via MANY
    # distinct channels (Peppa Pig Gets Grounded, L Bergeron, Jeremiahmattick49...)
    # so per-channel blocking can't keep up; the title marker is unambiguous
    # (no genuine witnessed clip is titled "gets grounded" / "misbehaves at"):
    r"|\bgets\s+grounded\b|\bmisbehaves\s+at\b"
    # QA 2026-06-26: the dailymotion 'related' snowball keeps re-mining the Gordon
    # Ramsay reality-TV cluster (Kitchen Nightmares / Hell's Kitchen / Hotel Hell /
    # MasterChef -- ~50 hits, 0.7% of corpus). Real people + genuine-sounding heat,
    # so the LLM scores them witnessed (its scripted_media def only covers scripted
    # FICTION, not produced reality TV); the show brand is the unambiguous signal --
    # no candid clip is titled "Kitchen Nightmares". NB deliberately NOT blocking
    # "ellen"/"teen mom": "Ellen Dance Dare behind cops" is genuine candid public
    # footage, and "TEEN MOM ... ON OMEGLE" is an Omegle clip, not the show.
    r"|\bkitchen\s*nightmare|\bhell'?s\s*kitchen\b|\bhotel\s*hell\b"
    r"|\bmaster\s*chef\b|\bgordon\s*ramsay\b"
    # QA 2026-06-30: recurring staged/scripted brands surfaced repeatedly in
    # random spot-checks -- scripted morality-skit channels (DramatizeMe, in the
    # Dhar-Mann mold), produced prank-call / kissing-prank brands (Ownage Pranks,
    # PrankInvasion), foreign-language soap operas (telenovela/teleserye), and the
    # "try not to laugh" compilation format. High-precision brand/format markers;
    # no genuine candid clip carries these.
    r"|\bdramatize\s*me\b|\bownage\s*pranks?\b|\bprank\s*invasion\b"
    r"|\bkissing\s*prank|\btelenovela\b|\bteleserye\b|\btry\s*not\s*to\s*laugh\b",
    _re.IGNORECASE)


def _junk_title(cand: dict) -> bool:
    t = cand.get("title") or ""
    return bool(_TITLE_SKIP.search(t))


# Channel blocklist (QA 2026-06-18): a small set of channels that are pure
# scripted/produced-media or compilation re-uploaders (>=2 spot-check defects,
# 0 genuine) -- teleserye/soap channels, Dhar Mann Studio, dashcam/fails
# compilation channels, prank/clickbait farms. Blocking by *channel* catches
# produced media that evades the title filter (generic or foreign-language
# titles). Loaded from config/channel_blocklist.json so it's auditable/editable
# without a code change; every block is logged to data/channel_block_log.jsonl
# so any channel can be audited and its clips restored.
import os as _os
import json as _json


def _load_channel_block() -> set:
    try:
        with open(_os.path.join("config", "channel_blocklist.json")) as f:
            data = _json.load(f)
        return {str(c).strip().casefold() for c in data.get("channels", []) if str(c).strip()}
    except FileNotFoundError:
        return set()
    except Exception:
        log.exception("channel_blocklist load failed; running without channel block")
        return set()


_CHANNEL_BLOCK = _load_channel_block()
log.info("channel blocklist loaded: %d channels", len(_CHANNEL_BLOCK))


def _blocked_channel(cand: dict) -> bool:
    if not _CHANNEL_BLOCK:
        return False
    ch = (cand.get("channel") or "").strip().casefold()
    return bool(ch) and ch in _CHANNEL_BLOCK


def _log_channel_block(uid: str, channel, title) -> None:
    try:
        with open(_os.path.join("data", "channel_block_log.jsonl"), "a") as f:
            f.write(_json.dumps({"uid": uid, "channel": channel,
                                 "title": (title or "")[:200], "ts": time.time()}) + "\n")
    except Exception:
        pass


def _dedup_matches(matches: list[dict], window: float = 5.0) -> list[dict]:
    """Drop near-duplicate reactions (within `window` s) so clips don't overlap.
    Prefer LLM-detected (tier 5) over keyword when two collide."""
    ordered = sorted(matches, key=lambda m: (m["start"], 0 if m.get("tier") == 5 else 1))
    kept: list[dict] = []
    for m in ordered:
        if any(abs(m["start"] - k["start"]) <= window for k in kept):
            continue
        kept.append(m)
    kept.sort(key=lambda m: m["start"])
    return kept


def saved_window_count(saved) -> int:
    """Validate the clip-extractor return contract and return its row count."""
    if not isinstance(saved, list):
        raise TypeError(
            "clip extractor must return a list of saved-window records; "
            f"got {type(saved).__name__}"
        )
    return len(saved)


def resolve_llm_result(cfg, kw_detector, transcript, res):
    """Map a non-error LLMDetector result onto the routing tuple
    (matches, verdict, modality, agent, expand_queries, scene).

    Shared by the live loop (_detect_all) and the offline batch pipeline so the
    routing semantics can never drift apart. Verdict 'drop:*' = hard-negative.
    """
    if res.get("is_target") != "yes":
        # hard-negative (staged/scripted/spectacle/etc.) -> drop the whole video
        return [], "drop:" + str(res.get("hard_negative_type")), None, None, [], None
    agent = res.get("agent") or "human"
    expand = res.get("expand_queries", [])
    if res.get("modality") == "commentary":
        return res.get("matches", []), "none", "commentary", agent, [], res.get("scene")
    # Clip ONLY LLM-verified reactions. Keyword matches ("the fuck", "oh my god")
    # were previously unioned in after a merely VIDEO-level LLM verdict, which cut
    # clips at incidental profanity (the top phrase across the corpus was a generic
    # "the fuck" x415/1500). Keywords now serve only as a fallback when the LLM said
    # yes but none of its quotes could be located in the word stream.
    llm_matches = res.get("matches", [])
    if llm_matches:
        return _dedup_matches(llm_matches), "none", "witnessed", agent, expand, res.get("scene")
    kw_matches = kw_detector.detect(transcript) if cfg.get("llm", {}).get("keep_keywords", True) else []
    return _dedup_matches(kw_matches), "none", "witnessed", agent, expand, res.get("scene")


def _detect_all(cfg, kw_detector, llm_detector, transcript, uid, title=None):
    """Run keyword + LLM detection. Returns the routing tuple (see
    resolve_llm_result). LLM error -> keyword-only fallback (live loop only;
    the batch pipeline retries errors on its next run instead)."""
    if llm_detector is None or not cfg.get("llm", {}).get("enabled", False):
        return kw_detector.detect(transcript), "none", "witnessed", "human", [], None

    res = llm_detector.detect(transcript, title=title)
    if res.get("is_target") == "error":
        log.warning("LLM detector errored for %s; falling back to keyword-only", uid)
        return kw_detector.detect(transcript), "none", "witnessed", "human", [], None
    return resolve_llm_result(cfg, kw_detector, transcript, res)


def process_video(conn, cfg, transcriber, kw_detector, llm_detector, instr_detector,
                  cand: dict, query: str, category: str = None,
                  query_source: str = None):
    """Full pipeline for one candidate.

    Returns the number of reactions kept (>=0) for a candidate we actually
    transcribed, or None if it was skipped (filtered, or a download/transcription
    failure) -- None means "don't count toward query stats or the max-videos
    budget", so the loop seeks a real video.
    """
    uid = cand["uid"]
    scrape.save_metadata(cand, query, cfg, query_source=query_source)
    # provenance block stamped into every clip's metadata.json so the corpora are
    # never lumped: which platform, which query STRING, which query TYPE, which norm.
    prov = {
        "platform": cand.get("source"),
        "found_by_query": query,
        "query_source": query_source,   # seed/taxonomy/instructional/exploit/llm_expand/related/...
        "category": category,
    }

    if _too_long(cand, cfg):
        state.set_status(conn, uid, "skipped", skip_reason="duration_too_long")
        return None
    if _junk_title(cand):
        log.info("TITLE-SKIP %s: %r", uid, (cand.get("title") or "")[:80])
        state.set_status(conn, uid, "skipped", skip_reason="junk_title")
        return None
    if _blocked_channel(cand):
        log.info("CHANNEL-BLOCK %s: channel=%r %r", uid,
                 cand.get("channel"), (cand.get("title") or "")[:60])
        _log_channel_block(uid, cand.get("channel"), cand.get("title"))
        state.set_status(conn, uid, "skipped", skip_reason="blocked_channel")
        return None

    media = scrape.download(cand, cfg)
    if media is None:
        state.set_status(conn, uid, "skipped",
                         skip_reason="download_failed_filtered_or_unavailable")
        return None
    state.set_status(conn, uid, "downloaded")

    # GPU-FREE crawl (llm.deferred): transcription + detection run in the batch
    # pipeline (batch_pipeline.sh -> src/batch_transcribe -> src/batch_detect)
    # whenever the queue fills AND a big-enough GPU is free. The raw video is
    # kept until the batch decides hit/miss; the loop just keeps enumerating.
    if cfg.get("llm", {}).get("deferred", False):
        state.set_status(conn, uid, "pending_transcribe")
        return 0

    try:
        transcript = transcriber.transcribe(Path(media), uid)
    except Exception as e:
        log.exception("transcription failed for %s", uid)
        state.set_status(conn, uid, "error", error=str(e)[:500])
        scrape.purge_raw(uid, cfg)
        return None
    state.set_status(conn, uid, "transcribed")

    # instructional vein: didactic teach-a-norm videos (instr_* queries) -> a SEPARATE
    # corpus via the InstructionalDetector (demo span + educator explanation = label),
    # not the candid bystander-reaction pipeline.
    if instr_detector is not None and category and str(category).startswith("instr"):
        ires = instr_detector.detect(transcript, cand.get("title"))
        if ires.get("is_instructional") == "yes" and ires.get("demos"):
            n = clip_extract.save_instructional(Path(media), uid, ires, cfg, cand, category, prov=prov)
            state.finalize_instructional(conn, uid, n)
            return n if n else 0   # processed; not a candid hit (raw kept for re-clip)
        # not actually instructional -> fall through to the candid detector below

    matches, verdict, modality, agent, expand_queries, scene = _detect_all(
        cfg, kw_detector, llm_detector, transcript, uid, title=cand.get("title"))
    return finish_detection(conn, cfg, uid, cand, Path(media), query, category,
                            query_source, prov, matches, verdict, modality, agent,
                            expand_queries, scene)


def finish_detection(conn, cfg, uid: str, cand: dict, media, query: str,
                     category: str, query_source: str, prov: dict,
                     matches, verdict, modality, agent, expand_queries, scene) -> int:
    """Route a detection result: commentary -> text corpus, witnessed -> clips +
    negatives + snowball + expansion, drop/miss -> purge. Shared by the live loop
    and the offline batch pipeline (src/batch_detect). Returns reactions kept."""
    # scene = {scene_type(action/narration/mixed), n_people, violator_role, reactor_role,
    # severity, reaction_strength, n_speech_pauses} -> stamped into the clip metadata
    # so narration FPs are filterable and clips are rankable by interestingness.
    if scene:
        prov["scene"] = scene

    # Detector-cleared curated mundane twins become verified negatives rather
    # than ordinary misses. The main crawl is normally deferred, but retain the
    # same routing contract for un-deferred runs.
    if query_source == "null" and str(verdict).startswith("drop:"):
        # save_null_verified returns saved-window records, while the DB stores
        # a scalar count. Keep the conversion explicit so a list can never
        # reach SQLite's integer field and abort the deferred-detection batch.
        saved_null = []
        if media is not None:
            saved_null = clip_extract.save_null_verified(
                media, uid, cfg, url=cand.get("url"), category=category, prov=prov)
        n_null = saved_window_count(saved_null)
        state.finalize_null(conn, uid, n_null)
        if n_null > 0 and cfg.get("loop", {}).get("purge_raw_on_miss", True):
            scrape.purge_raw(uid, cfg)
        return n_null

    # Commentary always keeps its source video. The text label remains a
    # separate product, while later visual QA may recover a clean event clip.
    if matches and modality == "commentary":
        source_video = None
        if media is not None:
            try:
                source_video = clip_extract.retain_discussion_video(Path(media), uid, cfg)
            except (OSError, ValueError) as exc:
                # Never purge the only source copy when stable retention failed.
                log.error("could not retain commentary video %s: %s", uid, exc)
        clip_extract.save_discussion(
            uid, matches, cfg, cand, category, agent=agent, prov=prov,
            source_video=source_video,
        )
        state.finalize_discussion(conn, uid, agent=agent)
        log.info("DISCUSSION %s (%s, agent=%s): %d norm statements; video=%s",
                 uid, cand.get("source"), agent, len(matches), source_video)
        if source_video is not None and cfg["loop"].get("purge_raw_on_miss", True):
            # Removes a redundant raw copy/info sidecar. The stable retained
            # video lives outside purge_raw's target directory.
            scrape.purge_raw(uid, cfg)
        return 0  # processed, but not a witnessed-video hit

    if matches:
        saved = clip_extract.extract_clips(Path(media), uid, matches, cfg, cand.get("url"), agent=agent, prov=prov)
        if not saved:
            # all reaction groups dropped (e.g. truncated pre-roll) -> treat as a
            # miss: no snowball/expansion, purge the raw, count 0 reactions
            if cfg["loop"].get("purge_raw_on_miss", True):
                scrape.purge_raw(uid, cfg)
            state.finalize_video(conn, uid, 0)
            return 0
        # matched no-reaction negatives from elsewhere in this same video (anticipation task)
        clip_extract.sample_negatives(Path(media), uid, saved, cfg, cand.get("url"), prov=prov)
        state.record_reactions(conn, uid, saved)
        state.finalize_video(conn, uid, len(saved), modality="witnessed", agent=agent)
        # recommendation-snowball: a hit's related videos are likely hits in the
        # same neighborhood -> enqueue them so the crawl follows the content graph
        # instead of running out of search terms (Loubbrad/yt-scrape style).
        snowball_cfg = cfg.get("snowball", {})
        if (snowball_cfg.get("enabled", False)
                and snowball_cfg.get("dailymotion_enabled", True)
                and cand.get("source") == "dailymotion"):
            # the child inherits this hit's norm category so the whole snowball
            # branch stays traceable to a norm class (paper provenance)
            from .snowball_scope import audit_snowball_parent
            decision = audit_snowball_parent(
                title=cand.get("title") or "", agent=agent, scene=scene,
                reaction_count=len(saved))
            inserted = decision.allowed and state.add_query(
                conn, "dmrelated", cand["native_id"], source="related",
                priority=cfg["snowball"].get("priority", 1.5),
                category=category)
            state.record_snowball_proposal(
                conn, parent_video_id=uid, platform="dmrelated",
                related_query=cand["native_id"], category=category,
                decision=decision, inserted=inserted)
            if decision.allowed:
                log.info("snowball: ALLOW %s (%s, inserted=%s)",
                         uid, decision.reason, inserted)
            else:
                log.info("snowball: BLOCK %s (%s)", uid, decision.reason)
        elif (snowball_cfg.get("enabled", False)
              and snowball_cfg.get("reddit_enabled", True)
              and cand.get("source") == "reddit"):
            # Reddit snowball = subreddit discovery: a hit's crossposts reveal
            # OTHER norm-violation subreddits carrying the same video. We don't
            # refetch the video (crossposts are identical) -- we add the new subs
            # to the firehose to mine their full history. INSERT OR IGNORE makes
            # rediscovering known subs a cheap no-op.
            root = cand.get("crosspost_parent") or ("t3_" + cand["native_id"])
            for sub in sources.crosspost_subreddits(root, cfg):
                added = state.add_query(conn, "reddit", sub, source="related-sub",
                                        priority=cfg["snowball"].get("priority", 1.0),
                                        category=category)
                if added:
                    log.info("snowball: discovered new subreddit r/%s (from %s)", sub, uid)
        # LLM-driven query expansion: the scorer proposes new search terms grounded
        # in this hit. Dedup (INSERT OR IGNORE) + zero-hit pruning keep it from
        # degrading/looping; tagged source='llm_expand' for audit.
        if cfg.get("llm", {}).get("expand_queries", True):
            from .query_scope import audit_query_scope
            for q in (expand_queries or [])[:3]:
                decision = audit_query_scope(q, "llm_expand")
                inserted = decision.allowed and state.add_query(
                    conn, "dailymotion", q, source="llm_expand",
                    priority=1.0, category=category)
                state.record_query_proposal(
                    conn, platform="dailymotion", query=q,
                    source="llm_expand", decision=decision,
                    inserted=inserted, parent_video_id=uid, category=category)
                if not decision.allowed:
                    log.info("llm_expand: BLOCKED %s query %r (from %s)",
                             decision.reason, q, uid)
                elif inserted:
                    log.info("llm_expand: + dailymotion query %r (from %s)", q, uid)
        log.info("HIT %s (%s): %d reactions kept", uid, cand.get("source"), len(saved))
        return len(saved)

    # miss (or LLM-dropped hard-negative): transcript kept; drop the raw video
    if verdict.startswith("drop"):
        log.info("DROP %s (%s): LLM verdict %s", uid, cand.get("source"), verdict)
    if cfg["loop"].get("purge_raw_on_miss", True):
        scrape.purge_raw(uid, cfg)
    state.finalize_video(conn, uid, 0)
    return 0


def run(max_videos: int = 0, no_download: bool = False) -> None:
    cfg = state.load_config()
    setup_logging(cfg)
    state.ensure_dirs(cfg)
    conn = state.init_db(cfg)

    added = state.seed_from_config(conn)
    log.info("seeded queue from config: %s", added)
    terms = state.load_yaml("config/search_terms.yaml")
    vocab = terms.get("generator_vocab", {})

    deferred = cfg.get("llm", {}).get("deferred", False)
    transcriber = None
    if not no_download and not deferred:
        from .transcribe import Transcriber
        transcriber = Transcriber(cfg)
    kw_detector = detect_reactions.ReactionDetector(cfg)
    llm_detector = None
    instr_detector = None
    if deferred:
        log.info("DEFERRED mode: GPU-free crawl; transcription+detection handled "
                 "by the batch pipeline (batch_pipeline.sh)")
    elif cfg.get("llm", {}).get("enabled", False):
        from .llm_detect import LLMDetector, InstructionalDetector
        llm_detector = LLMDetector(cfg)
        instr_detector = InstructionalDetector(cfg)
        log.info("LLM-as-detector enabled (%s @ %s) + instructional detector",
                 cfg["llm"].get("model"), cfg["llm"].get("endpoint"))

    eval_every = cfg["loop"].get("eval_every", 20)
    processed = 0
    log.info("starting loop (max_videos=%s, no_download=%s)", max_videos or "inf", no_download)

    while not _STOP:
        # deferred mode: don't let raw video pile up unboundedly if the batch
        # pipeline can't get a GPU for a long stretch -- pause the crawl.
        if deferred:
            backlog = conn.execute(
                "SELECT COUNT(*) FROM seen_videos WHERE status IN "
                "('pending_transcribe','pending_detect')").fetchone()[0]
            if backlog >= cfg.get("batch", {}).get("max_pending", 3000):
                log.info("batch backlog %d >= max_pending; pausing crawl 10 min", backlog)
                time.sleep(600)
                continue
        q = state.next_query(conn, cfg)
        if q is None:
            cooldown_wait = state.query_cooldown_wait(conn, cfg)
            if cooldown_wait is not None:
                wait_sec = max(1.0, min(60.0, cooldown_wait))
                log.info("all eligible queries cooling down; next ready in %.1fs",
                         cooldown_wait)
                time.sleep(wait_sec)
                continue
            log.warning("no active queries left; running expansion")
            query_generator.evaluate_and_expand(conn, cfg, vocab)
            if state.next_query(conn, cfg) is None:
                log.error("still no queries; exiting")
                break
            continue
        platform, query, cursor = q["platform"], q["query"], q.get("cursor")
        category = q.get("category")
        query_source = q.get("source")   # query provenance type (seed/instructional/exploit/...)
        query_started = time.time()

        # YouTube needs a proxy/cookies; skip it if neither is available.
        if platform == "youtube" and not no_download and not scrape.proxy_ready(cfg):
            log.warning("youtube query '%s' skipped: no proxy/cookies available", query)
            state.record_query_run(
                conn, platform, query, started_at=query_started,
                candidates_returned=0, new_candidates=0,
                enqueued_candidates=0, skipped_candidates=0,
                cursor_before=cursor, cursor_after=cursor,
                run_status="source_unavailable", cfg=cfg,
            )
            continue

        candidates, next_cursor = sources.enumerate_candidates(platform, query, cfg, cursor)
        state.set_cursor(conn, platform, query, next_cursor)
        log.info("[%s] '%s' -> %d candidates (cursor=%s)", platform, query,
                 len(candidates), next_cursor)

        new_in_query = 0
        enqueued_in_query = 0
        skipped_in_query = 0
        for cand in candidates:
            if _STOP:
                break
            if not state.enumerate_video(conn, cand, query, platform, category,
                                         query_source=query_source):
                continue  # already seen -> never re-download
            new_in_query += 1

            if no_download:
                state.set_status(conn, cand["uid"], "enumerated")
                enqueued_in_query += 1
                continue

            n = process_video(conn, cfg, transcriber, kw_detector, llm_detector,
                              instr_detector, cand, query, category,
                              query_source=query_source)
            if n is None:
                skipped_in_query += 1
                continue  # skipped/failed -> doesn't count toward stats or budget
            enqueued_in_query += 1
            state.update_query_stats(conn, platform, query, processed_delta=1,
                                     hit_delta=1 if n > 0 else 0, reaction_delta=n)
            processed += 1

            if processed % eval_every == 0:
                query_generator.evaluate_and_expand(conn, cfg, vocab)
            if max_videos and processed >= max_videos:
                log.info("hit max_videos=%d; stopping", max_videos)
                _finish(conn)
                return

        run = state.record_query_run(
            conn, platform, query, started_at=query_started,
            candidates_returned=len(candidates), new_candidates=new_in_query,
            enqueued_candidates=enqueued_in_query,
            skipped_candidates=skipped_in_query,
            cursor_before=cursor, cursor_after=next_cursor,
            run_status="stopped" if _STOP else "complete", cfg=cfg,
        )
        log.info("[%s] '%s' done: %d/%d new, %d enqueued, %d skipped; "
                 "zero_streak=%d cooldown_until=%s",
                 platform, query, new_in_query, len(candidates),
                 enqueued_in_query, skipped_in_query,
                 run["zero_new_streak"], run["cooldown_until"])
        if no_download and not candidates:
            time.sleep(1)

    _finish(conn)


def _finish(conn) -> None:
    import json
    rep = state.report(conn)
    log.info("RUN SUMMARY:\n%s", json.dumps(rep, indent=2, default=str))
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Norm-violation video scraper loop")
    ap.add_argument("--max-videos", type=int, default=0,
                    help="stop after N processed videos (0 = run forever)")
    ap.add_argument("--no-download", action="store_true",
                    help="enumerate + record candidates only; no media/GPU")
    ap.add_argument("--report", action="store_true",
                    help="print state.db summary and exit")
    args = ap.parse_args()

    if args.report:
        state.print_report()
        return

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _handle_signal)
    run(max_videos=args.max_videos, no_download=args.no_download)


if __name__ == "__main__":
    main()
