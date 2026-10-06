# Social-Norm Video Corpus (draft dataset card)

A weakly supervised, multimodal corpus of **social-norm events**: short video intervals that show a
concrete social behavior, paired with evidence that the behavior is socially expected, disapproved of,
corrected, praised, or otherwise norm-relevant. Collected 2026 by the norm-scraper pipeline in this
repository. Status: **collected, not yet released**. This card is the draft for the Hugging Face dataset
repository; the data itself lives on our cluster (see "Where the data is" and "How to publish" below).

## Summary

| | |
|---|---|
| Snapshot | frozen physical inventory of 2026-08-11 |
| Retained videos | 121,461 |
| Weak-label rows | 156,774 (a video can carry several demonstrations, reactions or commentary statements) |
| Platforms | Reddit (via the Arctic Shift API, v.redd.it media), Dailymotion (official Data API), Odysee/LBRY (Lighthouse search). YouTube is disabled by default (bot-walls datacenter traffic). Roughly 85% of media is from Dailymotion. |
| Per-video artifacts | word-level WhisperX (large-v3) transcript + source URL for **every** processed video; for hits, the ~10 s pre-reaction clip (t−12 s … t+2 s), matched reaction phrase and timestamp, LLM/VLM scene labels, source-context labels, weak-label posteriors |
| Languages | mostly English; language is recorded per video |
| Identifier | `uid = "{source}__{native_id}"`, unique across platforms; used for dedup, filenames and source-disjoint splits |

### The three pillars

| Pillar | Evidence that the clip is norm-relevant | Retained media | Weak-label rows |
|---|---|---:|---:|
| **Witnessed** | an on-scene response to a behavior: objection, correction, intervention, protective response, other contemporaneous reaction | 22,204 | 32,955 |
| **Instructional** | an explicit statement of a social rule paired with a demonstration of the behavior (live action, role-play, animation, puppets, games, social experiments) | 52,299 | 64,888 |
| **Commentary** | a narrator, commentator, participant or observer expresses a normative stance toward a concrete event; visual supervision only when the event itself is present and localizable | 17,783 | 58,931 |
| **Negative controls** | detector-clean windows from the same sources, unmatched no-reaction windows, and null-verified "mundane twin" sources; 2,912 positive sources have a same-video matched negative window | 29,175 | — |

## What the labels mean

The true target, "a social-norm event occurred here", is latent. Labels are **weak**: they come from
tiered phrase and regex matching on transcripts, LLM scene labels, VLM frame labels, audio-event
detection and source-context classification, combined in an append-only, audit-gated program:

- 28 weak-signal mechanisms are registered; 10 passed manual audits on frozen, source-disjoint cohorts
  for narrowly declared uses (retrieval, review ranking, strict-route exclusion, pillar rerouting);
  18 failed to transfer and must abstain.
- 0 rules are approved for automatic acceptance; 0 rules automatically delete or reject media.
- A Snorkel-style label model over the labeling functions is designed (`docs/weak_supervision_lf_snorkel_implementation_v1.md`) but not yet fitted.

Known defects of first-pass labels, which any user must expect:

- a relevant behavior is discussed but not shown;
- a scene is visible but does not match the assigned norm or polarity;
- instructional clips contain explanation, slides or topical B-roll instead of a connected demonstration;
- commentary text is valid but the event footage is absent, badly localized, or contaminated by label-bearing narration or on-screen graphics;
- witnessed clips contain reactions to accidents, safety events, animals, spectacle or ordinary conflict without evidence tying the response to a social norm;
- the VLM confuses an affected participant, victim, camera operator, authority or prank target with an independent bystander;
- query lineage concentrates scripted series, compilations, news or police material.

## Intended tasks

From `docs/research_scoping_plan_v1.md`:

1. **Reaction prediction (R1).** Given the pre-reaction clip, predict reaction presence, responder role, strength (1–5) and latency. Matched within-video negatives are the defensible core.
2. **Selection function (S1).** What gets reacted to: reaction presence and strength against severity, head-count, norm domain, content type and platform.
3. **Enculturation channels (E1).** Does a model state the rule differently after seeing k clips from instruction, from observed sanction, or from commentary? The 5.5k explanation-only transcripts are the rule bank.
4. **Norm-aware agents (A1).** A reaction predictor as a social-cost model for a simulated agent.
5. **Counterfactual deltas (C1).** Matched pairs and minimal edits that flip the predicted reaction.

Leakage probes (text-only, audio-only, OCR-only) and difficulty matching on cheap visual statistics
should be reported beside every headline metric. Splits are source-disjoint, grouped by `uid` and, where
available, by channel or cluster.

## Files (proposed release layout)

```
metadata/videos.parquet          one row per uid: platform, native id, URL, duration, language, pillar, query lineage, retention state
transcripts/{shard}.parquet      word-level transcripts with timestamps, keyed by uid
labels/weak_labels.parquet       156,774 rows: uid, pillar, norm domain, polarity, signal ids, posteriors, audit status
labels/negative_controls.parquet matched and unmatched negative windows
clips/{pillar}/{uid}.mp4         ~10 s pre-reaction clips (gated; see below)
manifests/{task}_{split}.jsonl   task manifests built by experiments/
```

## Ethics, legal and access

- All media was public on its platform at collection time, but it shows **identifiable people**, often in
  unflattering moments. Do not release raw media openly. Recommended: publish `metadata/`, `transcripts/`,
  `labels/` and `manifests/` openly (CC BY 4.0 for our annotations), and **gate the clips** behind a Hugging
  Face access request with a stated research purpose and a no-redistribution, no-identification agreement.
- Keep source URLs so that platform takedowns and user deletions can be honoured; run a periodic
  re-check and remove clips whose source is gone.
- The underlying videos are not ours to relicense; the release covers our derived data (transcripts,
  labels, clips as fair-use research excerpts) and must say so.
- Platform terms: Reddit data arrives via Arctic Shift (Pushshift successor); Dailymotion and Odysee via
  their public APIs. Record this provenance per row.

## Where the data is

`sk3` (`skampere3.stanford.edu`): `/lfs/skampere3/0/alexspan/norm-scraper/data/` with `raw_video/`
(purged on miss), `transcripts/` (kept for all videos), `hits/{uid}/`, `metadata/` and `state.db`.
The project checkout on sk3 is symlinked at `~/norm-scraper`.

## How to publish (run on sk3, not from a laptop)

Do the export and upload on the cluster where the data already sits.

```bash
# 1. create the dataset repo (once), private or gated to start
hf repo create <org-or-user>/social-norm-video-corpus --repo-type dataset --private

# 2. export parquet tables from state.db and the ledgers (script to write: scripts/export_release.py)
#    -> release/metadata/, release/transcripts/, release/labels/, release/manifests/

# 3. upload the small, open tables first
hf upload <org-or-user>/social-norm-video-corpus release/metadata metadata --repo-type dataset
hf upload <org-or-user>/social-norm-video-corpus release/transcripts transcripts --repo-type dataset
hf upload <org-or-user>/social-norm-video-corpus release/labels labels --repo-type dataset
hf upload <org-or-user>/social-norm-video-corpus DATASET_CARD.md README.md --repo-type dataset

# 4. clips: shard into tar files of ~1 GB (webdataset style) and upload with the large-folder path
hf upload-large-folder <org-or-user>/social-norm-video-corpus release/clips --repo-type dataset

# 5. turn on gated access in the repo settings before making it public
```

Notes: `hf` is the Hugging Face CLI (`pip install -U huggingface_hub`); log in with `hf auth login`
using a write token on sk3. Keep every shard under 50 GB and prefer parquet or tar shards over millions
of small files. Record the snapshot date in the card and bump it on each re-export.

## Citation

Spangher et al., in preparation. Please contact alexspan@stanford.edu before using the media.
