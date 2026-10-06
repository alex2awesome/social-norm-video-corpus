# Norm-Scraper: Forward Plan
*(written 2026-06-11; lives in repo root + copied to sk3:~/norm-scraper/PLAN.md)*

## Where we are
- ~26k videos processed, ~5.2k hits → 4,070 witnessed dirs, 4,832 commentary, 1,182 instructional, 668 negatives.
- Scene/role annotation live since ~2026-06-05 (2,374/4,070 hits annotated); measured witnessed FP ≈ 20% (mostly solo-narration, news, scripted media).
- Full raw video IS kept for every witnessed hit (371 GB, 13 TB free on /lfs) — clips are derived views; we can always re-clip.

## Phase 0 — quality bundle (DEPLOYED 2026-06-11)
1. **Detector**: `scripted_media` hard-negative (full films / TV / gameplay-streams);
   pro wrestling → `sports_broadcast`; real hidden-camera pranks explicitly KEPT
   (violator_role=camera_person); `severity` (1-5 flagrancy) + `reaction_strength`
   (1-5 viscerality) added to scene block — the "interestingness" sort key.
2. **Hard rule**: witnessed + scene_type=narration + n_people≤1 → commentary.
3. **Clips**: reactions <22s apart merged into ONE clip; clips with <10s pre-roll dropped.
4. **Keyword matches no longer cut clips** when the LLM located its own quotes
   (generic "the fuck"/"oh my god" was the #1 clip trigger); keyword = fallback only.
5. **Title pre-filter** (zero bandwidth): full movie/film/episode, trailer, gameplay,
   GTA/FiveM/Minecraft/Roblox, WWE/AEW/wrestling, compilation.
6. **Content dedup**: same normalized title + duration ±2s → status='duplicate', skipped.
7. **llm_expand guard**: wrestling/gaming/NSFW expansion queries blocked at insert.
8. **Rumble OFF** (weight 0.0): search endpoint 403s + 79% commentary.

## Phase 1 — retroactive cleanup (COMPLETED 2026-06-11, `src/retro_cleanup.py`)
4,093/4,093 dirs in ~4.5h, zero LLM errors, zero missing transcripts. Tally:
recut 2,256 / title_filter 500 / dropped 531 (scripted_media 128, staged_prank 49,
other 65 + small buckets) / commentary rerouted 425 / duplicate 249 / annotate-only
132. Witnessed corpus: 4,093 → 2,449 hits (9,618 reactions, ~3.9/video, ≤6
clips/video after merging). Scene+severity coverage now 97% (was 58%).
Severity dist: 1:66 2:308 3:343 4:1118 5:517; strength: 2:337 3:201 4:1101 5:697.
**Gold slice: 1,531 clips at sev≥4 & strength≥4.** Everything removed sits in
`data/quarantine/<reason>/` (98 GB incl. pre_recut copies of all 2,517 re-cut
dirs) — DELETE ONLY AFTER USER SIGN-OFF.
Known residue (spot-checked in the gold slice): episodic TV ("Feriha #16",
"Cheaters" reality show) and dashcam SERIES ("Bad Drivers of Bristol #45") slip
past both filters — candidates for a `#\d+`-series-title heuristic or the Phase 3
curation pass. Severity/strength dist is top-heavy (LLM generosity) → treat as a
ranking signal, not calibrated absolute labels.

## Phase 2 — audio-event ("scream/gasp/commotion") channel  ★ user-endorsed
Transcript-keyword search is deaf to non-verbal reactions — exactly the most
visceral, "interesting" moments. Second detection channel, **Pass A LIVE 2026-06-12**:
- **Model chosen**: PANNs Cnn14_DecisionLevelMax SED (`panns_inference`, AudioSet
  527 classes, framewise ~10ms scores, ~2 GB VRAM, ckpt in /lfs panns_data).
  Chose over BEATs (higher mAP 0.486 but clip-level only, no localization) and
  CLAP (flexible zero-shot but uncalibrated; keep as optional 2nd opinion).
  25 target classes in 2 groups: vocal_reaction (Screaming, Shout, Yell, Gasp,
  Crying/sobbing, Children shouting, …) + commotion (Crowd, Slap/smack,
  Smash/crash, Shatter, Gunshot, Skidding, Siren, …). AudioSet labels are weak →
  scores are a RANKING signal, not ground truth (per-class AP ~0.4-0.6).
- **Calibration (2026-06-12)**: AudioSet sigmoids under-confident — gold hits
  peak 0.10-0.28 on scream classes vs <0.1 on quiet controls; threshold=0.12.
  Validated temporal alignment: events land ON the LLM reaction timestamps
  (Crowd@72s vs reactions [72,72]; Screaming@198 vs 203; Gunshot@190 vs 187)
  with semantically-right classes (Slap/smack through a fight video).
- **Pass A (DONE — `src.batch_audio_events`)**: every hit's raw (clips fallback)
  → `data/audio_events/<uid>.json` (full event timeline) + summary stamped into
  the hit's metadata.json as `audio_events: {scream_peak, commotion_peak,
  interest, n_events}`. Resumable (skip-if-json-exists); transient CUDA/OOM
  aborts without poisoning (uid stays todo). ~0.3 s/video on one B200.
- **Pass A2 (`src.audio_recover`, run 2026-06-12)**: vocal events ≥0.20 peak
  outside merge-window of any known reaction (and ≥10s in) → merged into ≤4
  windows/video → offline-vLLM judge (genuine spontaneous reaction vs music/
  crowd/ride/scripted/game audio; empty transcript explicitly NOT evidence
  against — non-verbal is the point) → "yes" windows cut as additional
  clip_N.mp4 into the EXISTING hit dir, reaction appended to metadata
  (tag=audio_event, audio_peak), and any matched-NEGATIVE clip overlapping the
  new reaction is DELETED (decontamination). Journal:
  data/audio_recover_journal.jsonl (resumable; parse errors retried).
  Census: 252 candidate windows / 181 videos (Gasp 75, Groan 66, Cheering 51,
  Crying 27, Screaming 18...).
- **Pass B (LIVE in batch_pipeline.sh as stage 3)**: after detect, score any
  new hits with batch_audio_events (skip-if-scored; WhisperX 14GB gate; skipped
  entirely when hits == scored). Periodically rerun audio_recover to harvest
  new non-verbal candidates from freshly scored hits.
- **Bonus (unlocked)**: rerank = severity × reaction_strength × audio interest.

## Phase 3 — curation & training prep
- "Interesting-first" view: rank witnessed clips by severity × reaction_strength
  (+ audio score when Phase 2 lands); export top decile as the gold demo set.
- Embedding dedup (near-duplicate clips across re-uploads the title dedup missed).
- Hard-negative mining: from the 668+ negative pools, pick calm windows whose
  embeddings are closest to positives.
- Negatives coverage fix: sample negatives from any non-compilation source
  (currently 240s duration cap → only 20% of hit videos have negatives).

## Full videos vs clips — decision
**Keep both (status quo, now deliberate).** The pipeline already keeps the complete
raw video for every witnessed hit (`purge_raw` only fires on miss/commentary), so
clips are cheap derived views, and Phase 1/2 re-clipping is possible precisely
because the raw is there. 371 GB for 11k videos is sustainable on /lfs (13 TB free);
revisit (e.g. re-encode raws to 480p) if we pass ~1 TB. Misses stay purged — keeping
all misses would be ~5× the storage for material we've judged irrelevant.

## Standing ops
- **GPU-FREE crawl + batch GPU pipeline (since 2026-06-11):** the crawl only
  enumerates + downloads (`status='pending_transcribe'`, raw kept; pauses if queue
  > batch.max_pending). `batch_pipeline.sh` (hourly cron, flock'd) fires when the
  queue ≥ batch.min_backlog (150) or the oldest item > max_age_hours (4):
  stage 1 = `src.batch_transcribe` (WhisperX, norm-scraper env, needs 14 GB);
  stage 2 = `src.batch_detect` (offline `vllm.LLM.chat`, ai_usage env, needs 100 GB).
  Each stage **waits for ANY big-enough GPU** (polls nvidia-smi 2-min intervals up
  to gpu_wait_minutes=50, then defers to the next cron) and releases it fully on
  exit. The resident :8017 vLLM server is RETIRED (`restart_vllm.sh` kept only for
  reverting via `llm.deferred: false`). Routing logic is shared with the live loop
  (`search_loop.finish_detection` / `resolve_llm_result`, detector
  `prepare()`/`finish()` halves) so batch and live can never drift.
  LLM errors stay `pending_detect` (retried next batch); hit/reaction query stats
  are applied at batch time (scheduler tolerates the lag).
- QUIET crawl (norm-scraper-quiet, GPU 6) runs OLD code — never redeploy/prune it.
- 15-min monitor protocol: bot-wall grep, corpora counts, negative spot-checks,
  cycle maintenance, fail2ban-safe SSH (one connection, never hammer).
- Re-enable rumble (0.35) only if its 403 block lifts; consider bumping youtube
  above 0.45 if the android client keeps avoiding bot-walls.
