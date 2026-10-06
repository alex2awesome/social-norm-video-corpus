# Norm-Violation Video Scraper

The corpus this pipeline builds is described in [`DATASET_CARD.md`](DATASET_CARD.md) (what it is, the
three evidence pillars, label semantics, intended tasks, access policy, and how to publish it from the
cluster). This is an open mentorship project; see <https://www.alexander-spangher.com/wishlist.html>.

For the current research goals, corpus snapshot, audit findings, implemented
weak signals, and remaining roadmap, see
[`docs/project_goals_progress_and_remaining_work_20260817.md`](docs/project_goals_progress_and_remaining_work_20260817.md).
The standardized labeling-function interface and shadow label model that
implement that roadmap's Priorities 1–3 are documented in
[`docs/weak_supervision_lf_snorkel_implementation_v1.md`](docs/weak_supervision_lf_snorkel_implementation_v1.md).

Continuously mines public video platforms for clips containing audible
**bystander reactions to social-norm violations**. For every video processed it
saves a full word-level transcript + URL; for "hits" (videos containing a matched
reaction phrase) it also saves the ~10 s pre-reaction clip and matched-phrase
metadata. Built for `sk3` (skampere3.stanford.edu, 8× B200).

## Key finding (2026-06-01): platforms, not proxies

YouTube **bot-walls datacenter proxies** ("Sign in to confirm you're not a bot")
on every player client, while sk3's own IP downloads YouTube fine. Rather than
buy residential proxies, we pivoted to platforms that **don't bot-wall** and have
**free search APIs**. All of these work **direct from sk3, no proxy, no cookies**:

| Platform | Enumeration | Download | Status |
|---|---|---|---|
| **Reddit** | [Arctic Shift API](https://arctic-shift.photon-reddit.com) (Pushshift successor, no auth) → v.redd.it DASH | yt-dlp | ✅ primary |
| **Dailymotion** | official Data API (no key) | yt-dlp | ✅ |
| **Odysee/LBRY** | Lighthouse search API (no key) | yt-dlp | ✅ (per-item flaky) |
| **YouTube** | `ytsearchN` | yt-dlp (needs proxy/cookies) | ⚠️ disabled by default |

Reddit needs no API approval (the Nov-2025 Responsible Builder Policy gates the
*official* API; Arctic Shift serves the same dump data over HTTP). For bulk scale,
the per-subreddit `.zst` dumps on academictorrents are the offline fallback.

## Pipeline

```
queue (platform,query) ─▶ sources.enumerate_candidates   (Arctic Shift / DM API / Lighthouse / ytsearch)
                       ─▶ dedupe vs state.db             (never re-download a uid)
                       ─▶ duration pre-filter            (metadata, no bandwidth)
                       ─▶ scrape.download                (yt-dlp, direct; uid-named)
                       ─▶ transcribe.Transcriber         (WhisperX large-v3, word timestamps) ── ALWAYS saved
                       ─▶ detect_reactions               (tiered phrase + regex matcher)
                       ─▶ if hit: clip_extract (t-12s..t+2s) ; else purge raw
                       ─▶ stats + every N: query_generator (exploit/explore per platform)
```

Each candidate has a globally-unique `uid = "{source}__{native_id}"` used for
dedup, filenames, and `data/hits/{uid}/`.

## Layout

```
config/   search_terms.yaml (per-platform sources) · reaction_phrases.yaml · settings.yaml
src/      search_loop.py · sources.py · query_generator.py · scrape.py
          transcribe.py · detect_reactions.py · clip_extract.py · state.py
data/     raw_video/ (purged on miss) · transcripts/ (KEEP all) · hits/ · metadata/ · state.db
logs/     run_{date}.log
tests/    test_detect.py (offline) · test_sources_e2e.py (network+GPU)
```

## Deployment (sk3)

`~` on sk3 is AFS (token-expiring, small). The project + data live on local
scratch, with a convenience symlink:

```
/lfs/skampere3/0/alexspan/norm-scraper
~/norm-scraper -> /lfs/skampere3/0/alexspan/norm-scraper
```

`run.sh` activates the env with `HOME=/lfs/...` (AFS home breaks conda) and `cd`s
to the project before exec'ing — always invoke via `./run.sh <cmd>`.

### Environment

B200 is Blackwell (sm_100), needs a **CUDA 12.8** torch build (installed:
torch 2.8.0+cu128). WhisperX + ffmpeg are in the `norm-scraper` conda env.
GPU is auto-selected at startup (`transcribe.device_index: auto`) — sk3 is shared
and the free GPU moves around; set an int to pin.

## Usage

```bash
cd /lfs/skampere3/0/alexspan/norm-scraper

# one-time: init DB + seed queue from config sources
./run.sh python -m src.state

# offline matcher tests (no GPU/network)
./run.sh python -m tests.test_detect

# multi-platform end-to-end smoke test (network + GPU)
./run.sh python -m tests.test_sources_e2e

# bounded real run
./run.sh python -u -m src.search_loop --max-videos 20

# run forever under tmux
tmux new -s scraper
./run.sh python -u -m src.search_loop

# state.db summary (top queries + phrase frequencies + per-source breakdown)
./run.sh python -m src.search_loop --report
```

The loop is **idempotent and resumable**: all progress is in `state.db`
(including per-target pagination cursors); restarting skips anything already done.

## Configuring sources

`config/search_terms.yaml` → `sources:` block. Each platform has `enabled` plus
either `queries` (free text, expanded by the query generator) or `subreddits`
(Reddit, fixed). YouTube ships `enabled: false`; flip it on only with residential
proxies and/or a throwaway-account `cookies_file` (see `config/settings.yaml`
`network:`), then it routes through the rotating proxy pool.

## Reaction phrase tiers

See `config/reaction_phrases.yaml`. Tier 1 = direct censure ("that's so rude"),
Tier 2 = exclamatory surprise, Tier 3 = norm-naming regex ("we don't ___ at the
table"), Tier 4 = vocatives (kept only when co-occurring with Tier 1/2 within 5 s).
"oh my god" and profanity are gated/tagged. Each match stores
`{phrase, tier, tag, start, end, speaker, context}`.

## Notes / caveats

- Keep `data/transcripts/` for **every** video (hits + misses) — negatives are
  needed for the downstream norm-adherence work.
- Odysee returns no duration (can't pre-filter length) and is per-item flaky;
  failures are skipped gracefully.
- Bandwidth is free (direct from sk3), so full-video download is fine. If you ever
  switch back to paid residential proxies, add an audio-first pass (download
  `bestaudio` for detection, full video only for hits) to cut cost ~17×.
- Diarization (pyannote) is off by default; set `transcribe.diarize: true` + a
  `HF_TOKEN` to enable speaker labels. (The torchcodec warning at import is benign.)
```
