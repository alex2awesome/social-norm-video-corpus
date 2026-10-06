# Audit-gated rollout for weak supervision

Date: 2026-07-21

## Non-negotiable release rule

Production stays frozen until the relevant audit gate passes. Every changed
output means every datapoint whose accept/reject status, clip boundary, norm, or
polarity would differ under the candidate rule. Each such datapoint must have a
versioned VLM judgment, and a person must inspect the complete VLM output and its
evidence frames. Missing judgments, invalid responses, and `uncertain` judgments
fail closed.

The audit database is separate from the collection database. Audit scripts may
read videos and transcripts and write frames, manifests, and judgments under
`data/visual_audit/`; they must not edit `metadata.json`, `state.db`, routing code,
or extracted clips.

## Gates

1. **Frame/transcript mechanics.** Every selected clip must yield chronological
   frames spanning the interval, a stable frame-manifest hash, aligned dialogue,
   and a ledger row. Manually inspect every item for temporal coverage.
2. **VLM calibration.** Run the exact versioned prompt with the intended closed
   VLM. Manually review every response, including rejects and uncertain cases.
   Any invalid structured response, unsupported evidence frame, hallucinated
   action, or systematically missed medium fails the gate and creates a new
   rubric version.
3. **Complete affected-output audit.** Run the candidate rule in shadow mode.
   Audit 100% of the symmetric difference from current behavior, not a sample.
   Report false accepts, false rejects, uncertain cases, and results by category,
   polarity, query family, source, and medium.
4. **Production diff audit.** Review the exact code/config diff and reproduce its
   shadow output from a frozen manifest. Production passes only when every changed
   output has a reviewed judgment and there are zero unexplained false accepts.
5. **Post-deployment audit.** Audit every new accepted datapoint for the first
   collection interval. Continue hourly cohorts; automatically stop promotion if
   precision regresses or the input distribution changes.

Passing an earlier gate never implies a later gate passed.

## Current gate state

| Gate | Status | Evidence |
|---|---|---|
| Corpus inventory | Passed | Latest scan: 40,031 instructional demo records; 28,935 have clip files and 11,096 currently have no auditable clip |
| Rubric v1 mechanics | Failed | 8/8 reviewed; dialogue-dependent demos and trim-salvage were not represented correctly |
| Rubric v2 mechanics | Superseded | It handled aligned dialogue and trim salvage but did not represent audible educator/narrator label leakage |
| Rubric v3 mechanics | Passed | 8/8 reviewed; 3 accepted demos and 5 evidence-backed rejects; pure, mixed, and explanation-only composition validated |
| Instructional v4 manual evidence | In progress | 71 explanation clips: 0 clean, 5 usable only after semantic relabel/repair, 66 rejects; 24 non-explanation clips: 1 clean, 6 repairable, 11 rejects, 6 uncertain |
| Repair mechanics gate | Passed; production gate still closed | 4 temporal windows were frame-reviewed (3 pass, 1 reject); later exact hashed MP4 audits produced 2 passes and 2 rejects; both proposed overlay crops failed substantively |
| Interactive GPT-5.6-sol visual review | Available | Local Codex is authenticated with ChatGPT and its explicit configuration sets `gpt-5.6-sol` with high reasoning. Interactive visual judgments are recorded as Sol manual audits. |
| Unattended sk3 GPT-5.6-sol execution | Blocked | sk3 has no configured Platform API key or Codex access token/CLI session for scheduled jobs; local ChatGPT login is not inherited by the remote cron environment. |
| Instructional full audit | Not started | 28,799 clip-bearing records remain without an instructional-v4 judgment, plus disposition/recovery of 11,096 records without clips |
| Instructional production rule | Frozen | No live changes authorized by the audit evidence yet |
| Witnessed expanded manual audit | In progress | 48 total reviewed: 0 strict witnessed accepts, 7 instructional reroutes, 40 rejects, 1 uncertain; matched related/taxonomy cohorts were both 0/8 strict |
| Commentary expanded manual audit | In progress | 62 total reviewed: 9 clean accepts, 26 relabel accepts, 27 rejects; matched instructional/commentary query-source cohorts were 8/12 and 5/12 usable with full context |
| Commentary source retention | Active | Future sources are retained under `data/discussion_video/`; 425 surviving raw sources adopted, 12-source redownload canary manually passed storage integrity, Dailymotion/Rumble backfill running; source retention does not confer a visual label |

The two manual mechanics cohorts are explicitly stored under a model name that
does not claim to be GPT-5.6-sol. They validate the pipeline and rubric, not the
closed model.

### Missing-clip recovery audit

Of the 11,070 current no-clip records, 9,859 have invalid timestamps. Raw video
is still present for 9,667 of those; the other 1,211 records have valid spans but
no retained raw source. A shadow ordered-quote locator produced:

| Recovery confidence | Records |
|---|---:|
| Exact and one ordered candidate | 340 |
| Exact but multiple ordered candidates | 335 |
| Prefix match, one candidate | 17 |
| Prefix match, multiple candidates | 16 |
| Unresolved | 8,959 |

The first 24 exact/unique candidates were manually inspected. All 24 aligned to
the quoted transcript, but only 3 were valid demos, 1 was uncertain, and 20 were
rejects. Therefore ordered quote recovery is allowed only to create audit
candidates; it must never bypass the visual keep gate. No source metadata or clip
has been repaired yet.

### Search-query audit

Joining the 84 previously reviewed instructional clips to their originating
queries gave these observed rates:

| Query family | Reviewed | Valid demos |
|---|---:|---:|
| Scene-oriented in the original 108-clip audit | 7 | 5 (71.4%) |
| Targeted scene-oriented follow-up | 24 | 5 clean + 1 trim-salvageable |
| Scene-oriented combined | 31 | 10 clean (32.3%) + 1 trim-salvageable |
| Generic explanatory (`how to`, `lesson`, `training`, `explained`, etc.) | 70 | 10 (14.3%) |
| Neither family | 33 | 12 (36.4%) |

The targeted cohort disproved the blanket scene-keyword priority hypothesis.
Bare `scenario` yielded 1/9 clean demos; `role play` yielded 0/6 clean and 1/6
trim-salvageable. `social story` (2/4 clean) and `what would you do` (1/1) remain
promising but under-sampled. Generic terms are still low-yield, but the sample is
too small to deactivate individual queries. Title/query token overlap was tested
and was not predictive, so it is explicitly not a candidate filter.

## Instructional v4 acceptance contract

A usable demo requires all of the following:

- a social norm rather than a technical procedure, generic safety rule, animal
  behavior, medical advice, or self-improvement tip;
- an agent visibly performing, attempting, or refraining from behavior in a
  social context;
- support for the proposed norm from the combined temporal frames and aligned
  dialogue, without relying on title or narrator assertion alone;
- an observed polarity of `violation`, `correct`, or `contrast`;
- no answer-revealing text in the usable interval.

All media are valid. A dialogue act counts when visible participants are engaged
and the aligned utterance itself performs the act. A removable title card or
educator/narrator explanation yields `accept_after_trim` only when audited clean
bounds retain a complete scene. Persistent answer text or normative narration
rejects the clip. In-character dialogue remains part of the demonstrated
behavior, not narration leakage. Proposed `explanation` polarity is treated as
unlabeled scene polarity and must be resolved by the audit.

Ordinary subtitles that transcribe in-character dialogue are allowed because
they are part of the enacted exchange. Explanatory captions, normative answer
cards, and narration that supplies the label are not. A fictional classroom,
panel, or lecture is still explanation-only unless someone visibly performs the
norm-relevant behavior; placing a speaker inside an acted setting does not turn
a lecture into a demo.

### V4 polarity audit and repair evidence

The first v4 cohorts are calibration evidence, not corpus-wide precision
estimates. Across 71 reviewed `explanation` clips, none passed cleanly: five were
usable only after an explicit semantic relabel or separately audited repair and
66 were rejected. The dominant failures were talking heads, static cards,
generic B-roll, off-topic technical/health content, and narrator label leakage.

A matched source audit now finds 0/24 usable from current instructional
queries, 1/8 from `llm_expand`, and 2/8 from `related`; all three usable clips
needed both norm and polarity relabeling. These cells are too small for source
promotion or deactivation, and query provenance remains retrieval evidence only.
One additional cartoon contained a real bullying threat after an explicit
definition of bullying. Dense 0.25-second boundary review rejected it: the
threat was spoken while the bully was off-camera, and the bully appeared only
after the line ended, so no clean aligned recut existed. A broad domain match or
a demo somewhere in the clip is insufficient; the accepted interval itself must
contain the complete aligned social action without the answer leak.

Among 24 balanced `violation`/`correct`/`contrast` clips, one passed cleanly,
six passed with a concrete audited relabel or trim, eleven were rejected, and
six required a rendered repair audit. Four long candidates were recut to
quote-local windows and reviewed frame-by-frame. Three recuts retained complete
social demos and one failed because the norm-carrying adult remained off camera.
Therefore `tighten_to_demo` is now a supported *candidate repair*, but the
original item does not become positive until its rendered recut passes the
separate repair ledger. The observed 3/4 success rate is not permission to recut
or accept the rest of the corpus automatically.

Two persistent-overlay candidates were cropped and reviewed. Both crops removed
the text mechanically, but neither output qualified: one remained a reporter
describing an apology rather than showing it, and the other was a codified
driving-test hand signal rather than a tacit social norm. `crop_label_overlay`
therefore remains unsupported and is excluded from the candidate policy.

### Instructional expansion v2 and exact-artifact correction

A further 24-item v4 cohort produced 1 clean accept, 4 semantic relabel accepts,
17 rejects, and 2 transform candidates. Both candidates were then materialized
as exact MP4s and audited. The 29.229-second retail-support recut passed after
norm/polarity relabeling. Muting the Tea Consent animation alone failed because a
crossed-out tea answer card remained; a chained 11.000-second mute+trim repair
removed both audio and the answer card and passed. File SHA-256, duration, audio
stream count, and repaired frame-manifest hash are stored in the repair batch.

This audit also found a source-ranking difference: 6/8 `retro_instr_scan`
candidates were usable after repairs, versus 1/16 current instructional-query
candidates. The result supports a shadow rank-up for retro-scanned sources and a
rank-down for generic instructional queries, but is too small to deactivate any
query. Every candidate still requires the same visual keep gate.

A matched 12-versus-12 confirmation cohort strengthened that result. Seven of
12 fresh `retro_instr_scan` source clips passed the full v4 keep gate (one clean
and six only after explicit norm/polarity relabeling), four failed, and one long
source required an exact recut. The rendered 66.000-second recut preserved the
boundary-setting dialogue but showed almost entirely a car door and only part of
the seated listener; because the active speaker never appeared, the repaired
artifact failed. All 12 current generic-instructional-query clips failed, mostly
for no visual demo, off-topic procedural/safety content, and visible answer
leakage. Combined with expansion v2, the observed source-usable rates are 13/20
for `retro_instr_scan` and 1/28 for current generic instructional queries. This
supports ranking only; query deactivation and production routing remain frozen.

The 04:18 hourly interval then contained 296 new instructional demos, no new
witnessed clips, and six commentary records. A 12-clip instructional sample had
one visible demo and zero strict stored-label passes; the single demo was
escalated from the lightweight three-frame screen to a full 12-frame v4 audit.
That complete review accepted the scripted parent-child homework exchange only
after replacing generic “respect” with the concrete norm that parents should not
demean children who ask for academic help. All six commentary rows failed the
paired behavior-plus-stance gate. This interval reinforces fail-closed visual
review and label normalization; it does not authorize a production filter.

Those six commentary judgments used only the selected detector span. A later
matched full-context audit showed why that is insufficient: 8/12 fresh
instructional-query commentary records and 5/12 explicit commentary-query
records contained a grounded behavior-plus-stance pair once surrounding
transcript was inspected. Only 4/13 usable records supported their stored norm
without relabeling. Query provenance is therefore not a supported commentary
ranking feature; expanded context, two verbatim evidence quotes, and an explicit
normalized norm remain mandatory.

The 05:04 hourly pull re-sampled the same post-04:18 output pool because the
05:07 pipeline found only 114 pending transcripts and correctly skipped below
its 150-record/4-hour threshold. Twelve different instructional clips yielded
two norm-relevant visual demos: a clean enacted greeting and a public
intergroup-denigration speech that passed only after a concrete norm relabel. A
third visual dialogue was rejected as vocabulary practice with no tacit norm or
resolved polarity. Full-context review of all six commentary statements yielded
one relabel accept and five rejects; the failures were missing concrete conduct,
missing stance, nonhuman behavior, or a hypothetical event. No witnessed clips
were produced, so witnessed precision remains unmeasured for this interval.

The audit found that the original recut exporter persisted only windowed frames,
not the repaired MP4. Those four earlier recuts are now explicitly mechanics
calibration rather than production-ready artifacts. The exporter now persists
and hashes the exact recut MP4 and extracts review frames from that file.

### Multi-demo extraction audit

A fixed three-demos-per-video cap would remove 15,356 records across 5,256 source
videos and is not supported. In a two-video, 24-demo audit, all 12 clips from a
photosynthesis search collision failed at the video-level social-norm gate. The
12-demo WWYD source contained five clean demos, one trim-salvageable demo,
multiple explanation/interview rejects, and one overlapping duplicate pair.

The supported hypothesis is therefore: video-level social relevance, then
per-demo visual QA, then overlap/episode clustering. A global cap is rejected
because it would discard distinct valid scenes from high-yield sources.

### Full-corpus activity canary and dual-VLM replication (2026-07-24)

A frozen alphabetic-prefix canary sampled four low-, middle-, and high-activity
clips from each pillar. All 36 were reviewed blind from 12-frame sheets before
revealing scores, titles, norms, or query provenance; ten ambiguous items then
received temporal contact-sheet review. Strict current-pillar results were
instructional 1/12, witnessed 0/12, and commentary 0/12. Seven witnessed
rejects nevertheless contained potentially recoverable scripted, animated, or
hybrid visual scenes, reinforcing that rejection from witnessed must not mean
file deletion.

Activity was not a keep signal. The sole instructional pass was high activity,
but the same band also contained harmonica technique, medical graphics, and a
game interface. Witnessed recoverable scenes were most common in the middle
band, not the high band. Activity remains a diversity/scheduling feature only.

Qwen3-VL-8B-Instruct and GLM-4.6V-Flash completed 168/168 shadow records over
the exact cohort: label-blind literal-event v6, conditioned social/label-fit v5,
and witnessed-specific atomic v7c. The dual-v5 instructional intersection
selected the one pass, but is not promoted because the earlier 120-item
source-disjoint replication achieved only 76.9% precision. A dual-v5 witnessed
salvage intersection selected 3/3 recoverable scenes but found only 3/7, so it
is a manual reroute scheduler. Commentary dual-v5 selected one false positive.
Dual-v7c selected no witnessed item; the abstention is appropriate here but
cannot establish positive recall because the cohort contains no strict
witnessed positive.

The full 42,525 instructional, 30,576 witnessed, and 52,539 commentary
manifests are now undergoing non-destructive cheap shadow scoring. The scorer
is resumably sharded eight ways per pillar on low-priority CPU workers.
Commentary reports attempted, successfully decoded, and missing/unreadable
records separately because 25,492 statement records intentionally lack a
retained source video. The hourly monitor deduplicates base and shard ledgers
and alerts if any incomplete pillar has no live workers.

## Planned instructional audit order

1. Calibrate GPT-5.6-sol on the two mechanics cohorts and reconcile every
   disagreement with the manual records.
2. Audit at least 100 `explanation` clips because prior manual precision was
   1/22, then 100 balanced non-explanation clips.
3. Continue resumable batches until all available clips are judged.
4. Separately inspect records without clips to distinguish genuine
   extraction failures that can be recut from explanation-only records that are
   not visual datapoints.
5. Derive rules only from the completed ledger, run each rule in shadow mode,
   and manually inspect every changed output before touching production.

## Pillar-specific rule hypotheses

These are hypotheses to measure, not active rules.

### Instructional

- Require an audited visible demo; transcript grounding alone is insufficient.
- Separate educator explanation from observed scene polarity.
- Reject non-social procedural and off-topic content even when visually
  demonstrated.
- Preserve dialogue-dependent social acts using aligned transcripts.
- Trim removable label cards; reject persistent label leakage.
- Cluster overlapping or same-episode demos only after every proposed removal is
  audited; do not apply a fixed per-video cap.

### Witnessed

- Require the complete sequence `social action -> on-scene targeted reaction`.
- Require the reaction to be causally directed at the action, not generic affect.
- Keep only organic or genuine hidden-camera reactions; scripted scenes remain
  valid instructional material but not witnessed supervision.
- Exclude commentary/instructional query families and news/streamer narration.
- Verify that reaction-removal preserves the causative action while removing the
  reaction signal; audit every splice boundary.

The `witnessed_v1` contract now enforces all eight sequence/grounding fields in
code. `accept_after_splice` additionally requires organic or genuine
hidden-camera authenticity and ordered action-end/reaction-start bounds.
Scripted/animated rows can only use `reroute_instructional` when a visible action
supports the behavior label; that decision does not admit them to witnessed.

The 24-item expansion produced no strict witnessed positives and three
instructional reroutes. Across those 24, overlapping audited defects included
label mismatch (13), no visible social action (9), no on-scene reaction (8),
scripted/staged media (8), news/commentary (7), off-topic non-social events (7),
and generic affect (7). The complete causal-sequence and authenticity gate must
therefore remain mandatory; no single metadata field is supported as a positive
rule.

Matched eight-item `related` and `taxonomy` cohorts also produced zero strict
witnessed positives. Related yielded one scripted instructional reroute and
seven rejects; taxonomy yielded eight rejects. Across all reviewed rows, related
is now 0/23 strict and taxonomy 0/19 strict. A plausible cyclist/truck candidate
failed because the rider's reaction speech began before the alleged close-pass
action appeared, leaving no reaction-free action clip. Query families therefore
may not spawn related witnessed expansion until a manually calibrated strict
witnessed accept exists. Scripted reroutes still require a separate
instructional-v4 audit; the new reroute passed that second gate only after a
concrete norm relabel.

### Commentary

- Require a concrete actor, behavior, target/context, and explicit normative
  stance grounded in transcript text.
- Reject generic outrage, surprise, insult, event summary, and opinions with no
  conduct being judged.
- Keep commentary labels text-only unless a separate visual audit independently
  proves that the retained source contains the referenced same-event scene and
  the exact leak-free artifact passes review.
- Deduplicate repeated retellings or listicle items only after auditing every
  proposed merge/removal.

The `commentary_v2` contract accepts only transcript-grounded actor, concrete
behavior, target/shared context, normative stance, and supported norm. It also
supports `accept_after_relabel` when every semantic/evidence gate passes but the
stored norm is wrong or malformed; both verbatim evidence quotes plus concrete
normalized behavior and norm are mandatory. Titles, categories, agent metadata,
and queries are explicitly retrieval hints only.

The 24-item commentary expansion found 3 clean accepts, 13 accepts requiring a
specific norm relabel, and 8 rejects. Thus 16/24 grounded statements were usable,
but only 3/24 supported the stored norm as written. Detector `norm` cannot serve
as the final weak label without the v2 semantic audit and normalization.

The matched source audit added 24 rows: instructional-query commentary was 8/12
usable (2 clean, 6 relabeled) and explicit commentary-query material was 5/12
usable (2 clean, 3 relabeled). This rejects a provenance blacklist and instead
supports full-context semantic validation. Official sanctions, direct rules, or
recommendations count as a stance only when the same context grounds the actor,
conduct, and affected person or shared setting.

## Reproducibility artifacts

- `scripts/visual_audit_ledger.py`: corpus inventory and append-only judgments.
- `scripts/export_visual_audit_batch.py`: balanced/filtered sampling, adaptive
  temporal frames, transcript alignment, and hashed manifests.
- `scripts/run_openai_visual_audit.py`: versioned structured GPT vision rubric;
  resumes safely and refuses to run without an API key.
- `scripts/export_instructional_recut_audit.py`: renders exact proposed temporal
  repairs into a separate hashed MP4 batch without touching source clips.
- `scripts/export_instructional_audio_repair_audit.py`: removes explanatory
  audio, verifies stream counts, persists the exact MP4, and exports its frames.
- `scripts/compile_manual_instructional_repair_audit.py`: compiles complete
  post-repair manual reviews for the dedicated repair-judgment ledger.
- `scripts/run_openai_witnessed_audit.py`: versioned witnessed sequence,
  authenticity, reaction-causality, and splice-boundary rubric.
- `scripts/export_commentary_audit_batch.py`: balanced text audit export with
  detector-span flags and ±20-second timestamped transcript context.
- `scripts/run_openai_commentary_audit.py`: commentary v2 actor/action/target/
  stance gate with evidence-substring validation and audited relabel support.
- `scripts/compile_manual_witnessed_audit.py` and
  `scripts/compile_manual_commentary_audit.py`: compile complete human review
  tables through the exact candidate validators before ledger import.
- `scripts/report_audit_search_performance.py` and
  `data/visual_audit/search_source_audit_summary.json`: latest-judgment,
  pillar-scoped source/category yield used only for audited retrieval ranking.
- `scripts/audit_recover_instructional_spans.py`: shadow-only ordered quote
  recovery report; never cuts or edits corpus data.
- `scripts/export_recovered_span_audit.py`: frame exporter for manually checking
  recovered intervals.
- `audit_runs/instr_audit_mechanics_v1/`: failed v1 mechanics evidence.
- `audit_runs/instr_audit_mechanics_v2/`: v2 mechanics evidence and the defect
  that motivated audible-leak handling.
- `audit_runs/instr_audit_mechanics_v3/`: current passed mechanics evidence.
- `audit_runs/instr_span_recovery_audit_v1/`: 24-item manual recovery audit.
- `audit_runs/instr_scene_query_audit_v1/`: targeted 24-item search-term audit.
- `audit_runs/instr_multidemo_audit_v1/`: two-source, 24-demo cap/dedup audit.
- `audit_runs/witnessed_audit_mechanics_v1/`: eight-item manual mechanics audit;
  0 witnessed accepts, 3 instructional reroutes, 4 hard rejects, and 1 uncertain.
- `audit_runs/commentary_audit_mechanics_v1/`: failed binary-decision design
  evidence that motivated explicit audited relabeling.
- `audit_runs/commentary_audit_mechanics_v2/`: eight-item validator-checked
  manual mechanics audit; 2 accepts, 3 relabel accepts, and 3 rejects.
- `audit_runs/instr_explanation_audit_v2/`: 24-item instructional v4 explanation
  audit; 0 clean accepts, 2 repairable, and 22 rejects.
- `audit_runs/instr_explanation_source_instructional_v3/`,
  `audit_runs/instr_explanation_source_llm_expand_v1/`, and
  `audit_runs/instr_explanation_source_related_v1/`: matched 8/8/8 explanation
  source audit plus a 16-item current-instructional extension; 0/24, 1/8, and
  2/8 usable respectively, all usable only after norm/polarity relabeling;
  37/40 rejected.
- `audit_runs/instr_nonexplanation_audit_v1/`: 24-item balanced v4 audit; 1 clean
  accept, 6 repairable, 11 rejects, and 6 repair candidates.
- `audit_runs/instr_nonexplanation_recut_v1_rendered/`: four fully rendered
  temporal repairs; 3 accepted outputs and 1 reject.
- `audit_runs/instr_nonexplanation_crop_v1/`: two rendered overlay crops; both
  rejected after substantive review.
- `audit_runs/witnessed_expansion_v1/`: 24 clips and 384 temporal frames; 0
  witnessed accepts, 3 instructional reroutes, and 21 rejects.
- `audit_runs/commentary_expansion_v1/`: 24 transcript-context items; 3 clean
  accepts, 13 relabel accepts, and 8 rejects.
- `audit_runs/instructional_expansion_v2/`: 24 fresh v4 items; 1 clean accept,
  4 semantic relabel accepts, 17 rejects, and 2 exact-repair candidates.
- `audit_runs/instructional_expansion_v2_recut_exact/` and
  `audit_runs/instructional_expansion_v2_audio_trim_exact/`: exact hashed MP4
  repairs; both passed post-repair review.
- `audit_runs/20260721_141041_hourly/`: complete review of 24 fresh
  instructional clips (438 frames plus four exact-MP4 escalations) and all 23
  commentary statements from ten fresh videos with expanded context; one
  instructional clip and four commentary event statements survived. All
  outputs were reviewed and imported shadow-only.
- `audit_runs/instructional_querysource_retro_v1/` and
  `audit_runs/instructional_querysource_current_v1/`: matched 12-versus-12
  query-source confirmation; 7/12 versus 0/12 source clips passed.
- `audit_runs/instructional_querysource_retro_v1_recut_exact/`: exact hashed
  recut of the remaining retro candidate; rejected because the norm-carrying
  speaker remained off camera.
- `audit_runs/20260721_041859_hourly/` and
  `audit_runs/hourly_20260721_0418_candidate_v4/`: 12/6 lightweight hourly
  instructional/commentary audit plus the full v4 follow-up of its sole visual
  demo candidate.
- `audit_runs/20260721_050443_hourly/`,
  `audit_runs/hourly_20260721_0504_candidates_v4/`, and
  `audit_runs/hourly_20260721_0504_commentary_v2/`: complete 12-item visual
  screen, three full v4 follow-ups, and full-context review of all six
  commentary statements; two instructional demos and one commentary statement
  passed, with one relabel in each pillar.
- `audit_runs/witnessed_querysource_related_v2/` and
  `audit_runs/witnessed_querysource_taxonomy_v2/`: matched witnessed source
  cohorts; 0/16 strict accepts and one scripted instructional reroute.
- `audit_runs/witnessed_reroute_x7sjxnw_instructional_v4/`: independent v4
  review of that reroute; accepted only after a concrete sensitivity-norm relabel.
- `audit_runs/commentary_querysource_instructional_v1/` and
  `audit_runs/commentary_querysource_commentary_v1/`: matched full-context
  commentary cohorts; 8/12 versus 5/12 usable, with nine total relabels.
- `audit_runs/instr_contrast_retro_all_v1/`,
  `instr_contrast_related_all_v1/`, `instr_contrast_llm_expand_all_v1/`, and
  `instr_contrast_instructional_v1/`: 30 fully reviewed contrast sources; zero
  clean, two semantic relabels, 24 rejects, and four exact-repair candidates.
- `audit_runs/instr_contrast_*_exact_v*/`: seven valid, fully audited rendered
  artifacts; two accepted and five rejected. The accepted Good Samaritan chain
  preserves crop, audio-removal, trim, transcript, MP4-hash, and frame-hash
  provenance. The provenance-invalid `cropmute_trim2_exact_v1` batch has no
  judgment and is excluded from every acceptance count.
- `audit_runs/instr_violation_stratified_v1/` and
  `instr_violation_stratified_v1_replacements/`: corrected 24-item fresh,
  category-balanced `violation` sample; one clean source, four semantic
  relabels, 14 immediate rejects, and five exact-repair candidates. Three
  sampler duplicates were fully reaudited but excluded and replaced.
- `audit_runs/instr_violation_stratified_v1_recut_exact/` and
  `instr_violation_microaggression_silent_exact/`: six independently audited
  repair artifacts for five candidates; four accepted and two rejected.
- `audit_runs/20260721_1430_witnessed_dense/`: 24-source, 574-frame
  source-disjoint witnessed canary plus 234-frame recovery review; zero stored
  accepts, two `recover_strict` sources, four instructional-review reroutes, and
  18 rejects. All six non-reject outputs were manually reviewed.
- `audit_runs/20260721_1430_witnessed_recovery/exact_artifacts/`: exact hashed
  action-only MP4s for the two recovered organic sources, with the reaction
  clips held separately for provenance and excluded from model input.
- `audit_runs/20260721_1500_commentary_visual_recovery/`: retained-raw audit and
  a deterministic 12-source redownload canary. The unbiased canary reviewed 480
  overview/quote-window frames and 13 exact artifact iterations; four exact
  commentary visuals passed, one scripted clip entered instructional review,
  and seven sources remained text only. No corpus mutation was made.
- `scripts/backfill_commentary_videos.py`: locked, resumable source-video
  backfill with stable pseudo-random order, existing-raw adoption, ffprobe
  validation, SHA-256 provenance, and append-only attempt ledger.
- `audit_runs/20260721_1540_commentary_video_backfill_canary/`: 12 newly
  downloaded Dailymotion/Rumble sources, 288 frames, and complete manual storage-
  integrity judgments; 12/12 valid source media, without any automatic weak-label
  promotion.
- `scripts/build_distributed_youtube_manifest.py` and
  `scripts/distributed_youtube_backfill.py`: deterministic three-host YouTube
  recovery with a frozen manifest, one private credential per host, conservative
  disk/byte limits, authentication circuit breakers, ffprobe validation, hashes,
  atomic moves, and append-only node-local ledgers.
- `audit_runs/20260722_distributed_youtube_launch/`: three credential canaries
  plus a 9-source/108-frame post-launch audit; all audited downloads matched
  their source records. This is a storage-integrity approval only and does not
  promote any visual weak label.
- `scripts/hourly_collection_monitor.py` and
  `audit_runs/20260723_hourly_collection_monitor_first_run/`: append-only hourly
  health snapshots and a nonrepeating four-source review queue. The first queue
  passed 4/4 manual storage-integrity review; the monitor has no mutation or
  label-promotion authority.
- `audit_runs/20260723_commentary_event_localization_bottleneck/`: all 13
  previously unreviewed text-approved commentary sources with retained sk3
  video, 520 overview/target frames, four exact candidates, and two further
  sanitization iterations. One commentary visual passed, one scripted scene was
  routed to instructional review, and no production rule was changed.
- `audit_runs/20260724_instructional_rule_v1_replication_v1/`: 120 fresh,
  source-disjoint instructional clips reviewed over 1,440 frames, including
  exact follow-up of every ambiguity. Thirty were manually usable. The frozen
  GLM-4.6V-Flash plus Qwen3-14B rule selected 13, with 10 true positives after
  an append-only rubric correction (76.9% precision, 33.3% recall), so it
  failed promotion.
- `audit_runs/20260724_commentary_atomic_holdout_v1/`: 50 fresh, unique
  commentary targets and 202 six-second windows, all manually reviewed. The
  exact frozen localization rule selected none of 13 exact positives. A broader
  source-mining rule found two valid candidates among 22 recoverable sources,
  but 9.1% recall and only two selections are insufficient for promotion.
- `audit_runs/20260724_witnessed_atomic_holdout_v1/`: 80 fresh witnessed
  sources, excluding 1,291 previously reviewed UIDs, reviewed label-blind over
  960 frames and then with dense motion/audio evidence. One source passed the
  strict organic action-before-third-party-response contract and one remained
  provenance-uncertain. GLM-4.6V-Flash and Qwen3-VL-8B each completed all 80
  atomic audits. Their strict intersection selected two human rejects and
  missed the sole positive. The broad review intersection recovered the
  positive but at only 12.5% precision. All 21 clips selected by either model
  under any tested band were manually adjudicated; the rule remains
  shadow-only.
