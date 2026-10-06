# Visual weak-supervision audit and tightening rules

Date: 2026-07-21

## Deployment status

All rules in this document are audit hypotheses, not enabled production filters.
No crawler, detector, router, clipper, or corpus metadata has been changed from
these recommendations. A hypothesis may move into production only after every
output whose disposition it changes has a versioned visual judgment and every
model judgment has been manually reviewed. `uncertain` always fails closed into
the review queue; it never becomes a positive label.

## Target datapoints

The three pillars have different acceptance conditions and should not share one
binary `is_target` rule.

| Pillar | What is weak supervision? | Minimum accepted evidence |
|---|---|---|
| Witnessed | An on-scene person's reaction localizes and judges a violation | The clip contains the causally preceding social action, an on-scene reaction to that action, and a non-scripted participant/reaction |
| Instructional | An educator's explanation labels a demonstrated behavior | A visible temporal demonstration of the behavior linked to a grounded explanation; any medium is valid, including live action, animation, puppets, dolls, or an on-screen social exchange |
| Commentary | A speaker explicitly describes or judges social conduct | Text contains both a concrete social behavior and an explicit rule, criticism, praise, or recommended alternative; no visual-scene claim is made |

For instructional data, `scripted` is not a defect. The important distinction is
`visual demonstration` versus `explanation only`. A talking head can supply the
label but is not itself a demonstrated-behavior clip.

## Initial measurements

All video samples used three frames at 15%, 50%, and 85% of the extracted clip.
The baseline bundle is stratified across the corpus; the hourly bundle is from
the newest 90-minute collection window.

| Audit | Sample | Valid weak-label rate | Important detail |
|---|---:|---:|---|
| Starter instructional midpoints | 14 | 1/14 (7.1%) demos | Single frames; included four plainly off-topic disability-etiquette extractions |
| Stratified instructional baseline | 60 | 23/60 (38.3%) demos | Medium-agnostic: animation, puppets, and dolls count |
| Latest 90-minute instructional window | 24 | 1/24 (4.2%) demos | 460 eligible demo records arrived in the window; recent precision is much worse than the historical stratified sample |
| Latest one-hour instructional window (02:24 PT) | 24 | 2/24 (8.3%) demos | 179 eligible records; all 24 outputs manually labeled; no witnessed rows arrived and commentary was 1/2 valid |
| Stratified witnessed baseline | 30 | 7/30 (23.3%) valid organic weak labels | 17/30 contain some visible scene, but ten are scripted, mismatched, or authenticity-uncertain |

Across the 108 multi-frame instructional clips, 26/108 (24.1%) are valid demos.
Polarity is the clearest leak source:

| Polarity | Valid demos |
|---|---:|
| contrast | 4/5 (80.0%) |
| violation | 13/38 (34.2%) |
| correct | 8/33 (24.2%) |
| explanation | 1/31 (3.2%) |
| malformed (`don't`) | 0/1 |

Category estimates remain small and should not be treated as stable rankings.
Among categories with at least three sampled clips, the best observed were
`instr_autism_sel` (2/3), `instr_environmental` (2/4), `instr_autism_life`
(1/3), `instr_etiquette` (1/3), and `instr_family` (1/3). Zero-demo samples
included `instr_automobile` (0/4), `instr_consent` (0/6), `instr_digital`
(0/3), `instr_disability_etiquette` (0/4), and `instr_relationships` (0/6).
Query wording and polarity appear more predictive than category alone.

The fully auditable bundles and manual labels are in:

- `audit_runs/20260721_0137_baseline/`
- `audit_runs/20260721_014338_hourly/`
- `audit_runs/starter_14_manual_labels.tsv`

## Recommended gates

### 1. Witnessed pillar

Use a two-stage gate: transcript candidate generation followed by a short-video
VLM verdict on the full pre-reaction window.

Required transcript fields:

- `behavior_quote`: words identifying the actor and concrete conduct.
- `reaction_quote`: an on-scene sanction, objection, correction, or victim
  response. Generic surprise, profanity, excitement, or physical discomfort is
  not sufficient.
- `causal_target`: who/what the reaction targets.
- `norm`: an observable interpersonal rule, not an emotion or discourse state.
  Reject labels such as `surprise`, `confusion`, `physical discomfort`, and bare
  abstractions such as `respect` unless the concrete behavior is also stated.
- `source_authenticity`: `organic`, `hidden_camera_genuine_reaction`,
  `scripted_media`, `news_or_commentary`, or `uncertain`.

Required VLM verdict:

```text
event_visible: yes/no/uncertain
social_action_before_reaction: yes/no/uncertain
reaction_targets_action: yes/no/uncertain
authenticity: organic/hidden_camera_genuine/scripted/news/commentary/uncertain
label_supported_by_pixels: yes/no/uncertain
```

Accept only when the first three fields are `yes`, the label is supported, and
authenticity is `organic` or `hidden_camera_genuine`. Keep uncertainty in a QA
queue rather than treating it as positive.

Immediate deterministic rules:

- Never route a `comm_*` query/category into witnessed. The audit found
  `comm_debate_judgment`, `comm_aita`, and `comm_reaction_video` rows in hits.
- Keep the existing `instr_*` guard and apply the same pillar guard to all
  explicit query families.
- Drop witnessed candidates with clear movie/TV/animation watermarks, cinematic
  shot-reverse-shot editing, a news anchor, a streamer overlay, or a static
  audio/prank-call screen unless the VLM explicitly finds genuine hidden-camera
  participants.
- Do not accept a reaction quote whose only semantic content is `what the hell`,
  `oh my god`, `I am dizzy`, or equivalent generic affect.
- Do not expand or snowball from a hit until it passes the visual/authenticity
  gate. In the baseline, `related` produced 3/13 valid rows versus 4/12 from
  taxonomy queries.
- Treat `scene_type=action` only as a ranking feature, not an acceptance rule;
  it was valid in only 2/10 sampled rows.

### 2. Instructional pillar

Change the extraction schema so explanation and demonstration are separate:

```text
norm: actor + should/should-not + observable behavior + target/context
explanation_quote: grounded educator label
demo_start_quote / demo_end_quote: grounded candidate span
demo_medium: live_action/animation/puppet/doll/screen_social_exchange/other
demo_polarity: violation/correct/contrast
visual_demo_expected: yes/no
```

Candidate rules (frozen until the audit gate passes):

- `explanation` is not an observed behavior polarity. Explanation-only spans are
  annotations, not training clips. An explanation-labeled span may still be
  retained if the audited clip contains an actual demo; the auditor supplies its
  observed `violation`, `correct`, or `contrast` polarity.
- A demo must show an agent performing or refraining from an observable action
  in a social context. It can be silent or non-photorealistic; transcript form
  and realism are irrelevant.
- A visually grounded speech act counts: apology, refusal, insult, correction,
  praise, or similar dialogue is a demo when the frames show the participants in
  an active exchange and the aligned clip transcript contains the utterance.
  Narration over unrelated B-roll does not count.
- Require at least two of: in-character dialogue, two social agents, a framing
  transition (`watch`, `scenario`, `example`, `what should`), or an explicit
  before/after or correct/incorrect contrast. Silent demos can satisfy this only
  through the VLM.
- Reject technical procedures, product tutorials, medical advice, animal
  training, generic self-improvement, and physical safety unless the extracted
  behavior is an interpersonal/public norm. A useful structural test is whether
  the norm can be written as “A person should/should not do X toward Y/in shared
  context Z.”
- Reject underspecified or malformed norms (`rel`, `security`, `strength`,
  `curiosity`, `dim_weight`, `introduction`) and any polarity outside the enum.
- Require both grounded clip boundaries, a positive duration, and a maximum
  duration appropriate for one scene. Reject or review overlapping candidates.
- Do not impose a fixed per-video cap. The audited 12-demo WWYD source contained
  five clean, distinct demonstrations, while an equally large photosynthesis
  collision contained none. First gate the source for social relevance, then
  audit each demo, then cluster overlapping spans or repeated episodes; manually
  verify every proposed removal.
- Require query/category relevance, but do not use category as a substitute for
  visual verification.

VLM gate on the extracted span:

```text
visual_demo_present: yes/no/uncertain
medium: live_action/animation/puppet/doll/screen_social_exchange/static/talking_head/broll
agents_and_actions: concise description
norm_behavior_visible: yes/no/uncertain
polarity_supported: violation/correct/contrast/uncertain
off_topic: yes/no
```

Accept when `visual_demo_present=yes`, `norm_behavior_visible=yes`, and
`off_topic=no`. The explanation remains the weak label but should be outside the
eventual model input. If label-revealing cards or educator/narrator judgments
occur only before or after a complete demo, audit and store clean trim bounds
instead of discarding the scene. Reject clips whose normative answer text or
explanatory narration overlaps the whole usable scene. In-character dialogue is
part of the behavior and is not treated as educator/narrator leakage.

Do not blanket-mute instructional survivors. The exact-output audit must label
audio as diegetic behavior, non-label background, label-bearing explanation,
mixed/inseparable, absent, or uncertain. Preserve clean in-character audio when
it carries the act; mute only when the behavior remains complete without it;
reject required diegetic speech that is inseparable from educator narration.
The rendered artifact then receives a second audio/leakage audit. Commentary
visual repairs remain muted because commentary speech is the weak label;
witnessed pre-reaction splices preserve action audio while removing the later
reaction in both sound and pixels.

Any temporal, spatial, or audio transform is a repair candidate, not an accepted
datapoint. Render the exact output, hash its frames separately from the source,
and apply the full social-norm/demo/polarity/leak audit to the rendered result.
The first four quote-local recuts passed 3/4; the failed recut looked plausible
in the long source but left the norm-carrying adult entirely off camera. The
first two overlay crops passed the image operation but failed the substantive
gate (commentary rather than apology; codified driving procedure rather than a
tacit norm). Consequently only `tighten_to_demo` may create shadow repair
candidates, and neither recuts nor crops may be auto-accepted.

The next exact-output audit tightened this further: windowed frames are not
enough. A repair must persist the actual MP4 and store its file SHA-256, verified
duration, audio-stream count, and repaired frame-manifest hash. A mute-only Tea
Consent output failed because the final crossed-out tea answer remained visible;
the chained muted+trimmed 11.000-second MP4 passed. A separate 29.229-second
retail-support recut also passed after semantic relabeling. Intermediate failures
remain in the ledger and do not inherit the final artifact's acceptance.

A later matched query-source audit reviewed 12 `retro_instr_scan` and 12 current
generic-instructional candidates under the same v4 rubric. The source-clip pass
rates were 7/12 and 0/12 respectively. The only unresolved retro source was
rendered as an exact 66.000-second MP4, hash-verified, and rejected: its dialogue
described a valid confrontation, but the frame showed almost entirely a car door
and never showed the speaking participant. Across the expansion and matched
cohorts, usable rates are therefore 13/20 for `retro_instr_scan` and 1/28 for
current generic instructional queries. Treat this as evidence to rank sources,
not as permission to bypass visual QA or deactivate any query family.

In the 04:18 hourly sample, 1/12 current instructional clips contained a visible
demo and 0/12 passed with its stored label. The sole candidate was then expanded
to 12 temporal frames and accepted only after relabeling generic “respect” to a
specific anti-belittling norm for a parent responding to a child's homework
question. Six contemporaneous commentary rows all failed because none paired a
grounded concrete social behavior with an explicit normative stance.

That six-row screen used only the detector-selected span. In a later matched
audit with 20 seconds of transcript context, 8/12 instructional-query commentary
rows and 5/12 explicit commentary-query rows passed. Query provenance is not a
supported keep or ranking feature; the supported change is to require expanded
context before deciding whether both behavior and stance are grounded.

Do not apply a blanket priority to scene words. A targeted 24-clip follow-up found
only 5 clean demos plus 1 trim-salvageable demo: bare `scenario` was 1/9 and
`role play` was 0/6 clean (1/6 salvageable). `social story` (2/4 clean) and `what
would you do` (1/1) remain promising but under-sampled hypotheses. Generic
`explained`, `tips`, `lesson`, `how to`, and listicle queries remain low-yield,
but no term should be deactivated until all affected outputs are audited.

A matched audit of `explanation` clips sampled 24 from current instructional
queries and eight each from `llm_expand` and `related`; it found 0/24, 1/8, and
2/8 usable demos. None passed under the stored explanation label: all three
needed a concrete norm and resolved `correct` polarity. The 37 rejects included
talking heads, static story cards, unrelated B-roll, technical procedures, and
off-topic health/self-improvement material. These small cells do not support
source promotion or deactivation; query source remains retrieval metadata.

One rejected cartoon showed why “contains a demo” is not sufficient if the
accepted interval leaks the answer. It defined bullying before an actual threat;
dense 0.25-second review found the bully spoke off-camera and appeared only after
the threat ended, so removing the definition left no aligned visible speech act.
Ordinary in-character subtitles remain permissible, but explanatory captions,
definitions, and narration must be absent from the exact accepted interval.

### 3. Commentary pillar

Commentary is text supervision, not scene supervision. Require a paired semantic
structure:

```text
behavior: concrete act, actor, and affected person/context
normative_stance: criticism/praise/rule/recommended alternative
behavior_quote: grounded transcript span
stance_quote: grounded transcript span
```

The two quotes may be the same only if that sentence names both the behavior and
the stance. Reject generic exclamations, surprise, confusion, insults without a
target behavior, event description without judgment, and animal/accident/sports
reactions. A title can provide context for retrieval but cannot repair a quote
that does not itself ground the weak label.

Commentary v2 has two positive outcomes. `accept` requires the stored norm to be
supported. `accept_after_relabel` requires exactly the same actor, action,
target/context, stance, and verbatim-evidence gates, but is reserved for a
malformed, vague, or behavior-inverted stored norm; it must produce a concrete
normalized behavior and norm. Relabeling is therefore an audited repair, not a
way to infer missing conduct from titles or queries.

Keep `comm_*` material out of witnessed even if a candid-detector fallback sees
multiple voices. Full commentary source videos are now retained separately from
the text labels under `data/discussion_video/`; retention does not make a video a
visual positive. A source becomes a visual training item only after the referenced
same-event action is localized, rendered without the narrator/answer signal, and
the exact artifact passes the commentary visual audit. Scripted occurrences are
rerouted through the instructional visual gate.

### Witnessed v1 mechanics audit

The first eight-item, 16-frame-per-clip mechanics cohort passed extraction and
manual rubric coverage, but did not support any witnessed positive. Four rows
were rejected, three scripted/reality/prank scenes were rerouted as potential
instructional demonstrations, and one aftermath-only phone clip remained
uncertain. This is not a precision estimate; it is a deliberately small rubric
test.

The observed failure modes were concrete: detector-selected speech frequently
came from the source actor rather than a bystander, generic disclosure such as
“it's a prank” was not normative disapproval, news/crowd footage lacked a
localized causal action, and aftermath footage could not establish a clean
pre-reaction splice. Conversely, scripted scenes can visibly demonstrate the
proposed behavior and should be eligible for the instructional visual gate
without being mislabeled as organic witnessed evidence. Full judgments are in
`audit_runs/witnessed_audit_mechanics_v1/manual_review.tsv`.

The next 24-item/384-frame witnessed cohort produced zero strict positives,
three scripted instructional reroutes, and 21 rejects. Overlapping failure
counts were: label mismatch 13, no visible action 9, no on-scene reaction 8,
scripted/staged 8, news/commentary 7, off-topic 7, and generic affect 7. The keep
rule cannot be reduced to reaction lexicon confidence or `scene_type`; the full
visible action→targeted normative reaction→clean pre-reaction splice remains
mandatory.

Two fresh matched source cohorts added 16 clips and again produced zero strict
witnessed positives: `related` yielded one scripted instructional reroute and
seven rejects, while `taxonomy` yielded eight rejects. Related is now 0/23 strict
across reviewed clips and taxonomy is 0/19. A cyclist/truck clip that looked
plausible visually still failed because reaction speech began before the alleged
close-pass action appeared; there was no complete pre-reaction action to splice.
Do not expand a witnessed query family through related-video search until that
family has at least one manually verified strict witnessed calibration positive.
Every scripted reroute requires a fresh instructional-v4 review; the newly
rerouted separation-disclosure scene passed that second audit only after relabel.

### Commentary v2 mechanics audit

Commentary v1 failed its manual design review before use because it could only
accept or reject a candidate: that would discard fully grounded commentary when
the existing norm field was malformed. V2 added `accept_after_relabel` while
keeping all evidence invariants. On the same eight-item balanced mechanics
cohort, 2 were clean accepts, 3 were accepts requiring relabel, and 3 were
rejects. The rejected rows were an underspecified regret statement, a factual
news report without normative stance, and sports-performance criticism. This is
a rubric test, not a corpus precision estimate. The validated evidence is in
`audit_runs/commentary_audit_mechanics_v2/`.

In the next 24-item commentary cohort, 16 statements passed the actor/action/
target/stance gate, but 13 of those required norm relabeling; only three stored
norms were usable as written. Eight failed for missing stance, technical/legal
content, cross-item context, nonhuman harm, accident gratitude, or a bare insult.
Normalized behavior and norm are therefore mandatory outputs; detector `norm`
cannot be used as a training label without semantic QA.

The matched full-context source audit added 24 statements. Instructional-query
commentary was 8/12 usable (2 clean, 6 relabeled), while explicit
commentary-query material was 5/12 usable (2 clean, 3 relabeled). This rules out
source provenance as a keep/ranking shortcut. Direct rules, official sanctions,
criticism, praise, and recommendations may provide the stance, but only when
verbatim context also grounds the concrete actor, conduct, and affected person
or shared setting.

The 05:04 hourly cohort added six more full-context instructional-query
commentary judgments: one passed after relabeling unchecked character references
in an abuse case, while five failed for generic political insult without conduct,
affect without stance, nonhuman danger, underspecified anti-social behavior, or
a hypothetical shove that never occurred. Hypothetical or merely feared conduct
does not ground an observed violation, even when a speaker calls it dangerous.

### Instructional contrast-polarity audit

The remaining non-generic-source `contrast` candidates and an eight-item sample
from current instructional queries were reviewed exhaustively: 30 source clips
and 344 sampled frames. None was clean as stored. Two source clips were usable
only after semantic relabeling, 24 were rejected, and four were admitted only as
fail-closed repair candidates. Exact rendered repairs recovered two of those
four, producing an eventual 4/30 usable datapoints: 1/5 `retro_instr_scan`, 1/4
`related`, 1/13 `llm_expand`, and 1/8 current `instructional`.

`contrast` is therefore a retrieval hint, not a positive label. Acceptance
requires both behavioral poles to be visibly resolved in the exact audited
interval. A verbal comparison, lecture, narration over B-roll, or graphics is
not a contrast demo. If only one pole is enacted, it may pass only after an
explicit audited relabel to `violation` or `correct`.

Seven valid rendered artifacts were each reviewed independently: two passed
and five failed. The successful customer-service recut removed narration and
was relabeled from contrast to correct. The successful Good Samaritan artifact
required bottom-caption cropping, audio removal, and exact boundary trimming;
the final silent sequence visibly shows passersby ignoring an injured person
and a helper bandaging the injury. Three earlier versions of that same repair
were separately rejected for unrelated boundary shots. A fourth intermediate
manifest was excluded from the judgment ledger after the audit caught dropped
recursive audio-removal provenance.

That defect, plus a later dropped-transcript defect and a polarity-only relabel
validation defect, were fixed with regression tests before the accepted repair
was imported. Repair chains must now preserve recursive media transforms and
label-transcript provenance. A successful final transform never changes the
source clip's disposition and never erases failed intermediate attempts.

### Instructional violation-polarity audit

A category-balanced cohort initially selected 24 `violation` clips. Manual
cross-checking found that three had appeared in older audit batches because the
sampler excluded only judgments with the same rubric *and* model. All three
were still fully re-reviewed, but were removed from the fresh precision
denominator and replaced with three never-selected items. The exporter now
excludes any item with any source judgment or any prior batch assignment; ten
regression tests cover this and the repair-ledger invariants.

Among the corrected 24 fresh clips, only one passed cleanly as stored. Four
more passed after explicit norm or polarity relabeling, 14 failed immediately,
and five required exact repair attempts. Six rendered artifacts were audited
for those five candidates because the microaggression recut failed once for
offset audio and was rendered again without audio. Four final repaired outputs
passed and two artifacts failed. Eventual yield was therefore 9/24 (37.5%),
below the 50% monitoring threshold.

`violation` is consequently a retrieval stratum, not a keep rule. It performs
better than `explanation` and `contrast`, but still admits talking heads,
technical driving instruction, static hypothetical cards, unrelated security
graphics, off-screen alleged conduct, and extremely diffuse clips. Every kept
item still requires a visible actor/action/target or shared context, resolved
polarity, concrete normalized norm, and absence or removal of answer leakage.

Source provenance affected retrieval yield but not acceptance: retro scan was
7/12 eventually usable, current instructional queries were 2/11, and the lone
related item failed. Retro may be ranked above current instructional queries
for `violation` retrieval, but neither source can auto-admit a clip. The exact
repairs also established a new temporal failure class: a complaint or reaction
does not recover a violation whose action occurred before the clip.

### Audit-derived search ranking

`scripts/report_audit_search_performance.py` selects the latest manual judgment
per item, enforces the matching pillar, and reports source/category retrieval
yield without treating repaired artifacts as source passes. Its totals reconcile
exactly with coverage at that checkpoint: 218 instructional-v4, 48 witnessed-v1, and 81
commentary-v2 items.

After the 07:06 cohort, instructional source yield is 25/50 for
`retro_instr_scan`, 9/156 for current `instructional`, 2/23 for `llm_expand`,
and 2/13 for `related`. This
supports ranking retro higher and ranking generic current/LLM retrieval lower,
but not deactivation: every instructional source has at least one reviewed
positive. Category ranking is likewise provisional. Categories with at least
3/4 source passes may be ranked up, while five native instructional categories
with 0 source passes in 8–12 reviews may be ranked down; every item still faces
the same visual keep gate.

The expanded native-category evidence adds `instr_consent` to the ranking-only
down list: it now has 0/11 source passes. The other audited zero-yield native
categories are `instr_family` (0/18), `instr_crosscultural` (0/16),
`instr_automobile` (0/13), `instr_digital` (0/8), and
`instr_elder_intergen` (0/8). These are not deactivation rules.

Witnessed remains frozen for positive expansion because every reviewed source
cell is 0 strict: related 0/23, taxonomy 0/19, unclassified 0/4, and related-sub
0/2. Commentary taxonomy retrieval may be ranked up at 11/15 usable. Other
commentary sources remain active—17/38 instructional, 6/13 commentary, and 4/9
related—and none bypasses expanded-context actor/action/target/stance review.

### 06:13 post-batch hourly audit

The normal batch captured 223 rows and completed with 81 instructional routes,
11 commentary routes, 131 drops, one miss, and zero pending detections. The
post-batch 1.5-hour pool contained 389 instructional clips, nine commentary
videos, and no witnessed outputs. Every selected artifact was reviewed: 24
exact instructional MP4s at nine temporal positions and all 19 commentary
statements with expanded transcript context.

Only 2/24 instructional source clips passed, both after semantic relabeling: a
staged restaurant intervention about alcohol during pregnancy and a staged
racist-cabbie exchange. One news clip contained a real social-distancing demo
but leaked the answer through narration and lower thirds. An exact 296x256
right-panel crop with audio removed was rendered, hashed, and reviewed over 12
frames; it passed, making the eventual instructional yield 3/24 (12.5%). The
other 21 clips were talking heads, generic B-roll, or technical, medical,
fitness, acting, finance, software, marketing, and policy material without the
claimed social demonstration. This confirms that medium is not the rule: news,
animation, or staged footage may pass, but only when the exact retained pixels
show the labeled social behavior and any answer leakage is removed.

The commentary audit accepted 8/19 statements (one clean, seven after relabel)
and rejected 11. A later full-transcript dedup audit corrected the initial
three-scenario estimate: those eight statements resolve to four commentary
clusters. There is one crowding event, one alleged volatile-workplace event, one
concrete Wa battering report, and one aggregate critique explicitly referring to
two separate military incidents. Two market statements and two workplace
statements are duplicates. Of the four military statements, items 0 and 3 ground
the Wa occurrence, while items 2 and 4 judge the pair of incidents collectively;
they must not all be collapsed merely because they share an actor class and norm.
Commentary output
must therefore be clustered by source event and normalized behavior, retaining
only the strongest grounded behavior-plus-stance pair. Abstract labels such as
“hostile work environment,” “harassment,” or “human rights” fail unless nearby
transcript context supplies a concrete actor, act, target/shared context, and
explicit normative stance.

Two frozen clustering prompts were then tested on those eight accepted
statements. V1 merged all four military statements and got 2/3 source
partitions exactly right. V2 noticed multiplicity and emitted the correct total
of four clusters, but still mispartitioned the military source as `{0,2,3}` and
`{4}` instead of the transcript-supported `{0,3}` and `{2,4}`. It also scored
2/3 exact source partitions and is not production eligible. The audited rule is
therefore stricter than generic semantic deduplication: cluster exact
occurrences; do not attach commentary about a set of incidents to one
constituent incident merely because actor, action class, and norm match.

### Frozen instructional prompt comparison

Two transcript-only shadow prompts were run against the same 24 fully reviewed
source videos without changing the corpus. The production prompt emitted 155
demos. Shadow v2 emitted 12: one passed as-is, two were repair candidates, and
nine failed. Shadow v3 tightened the definition of a demo, prohibited explanation
polarity, required a single bounded event, and emitted 14: two passed as-is,
three were repair candidates, and nine failed. Raw precision was therefore 8.3%
for v2 and 14.3% for v3. Neither prompt is production eligible.

The v3 audit found that the model ignored its own requested time bounds. A
deterministic shadow contract now requires grounded start, end, and label quotes;
nonempty actor, behavior, target/shared context, and concrete norm fields; a
resolved `violation`, `correct`, or `contrast` polarity; at least two seconds of
context; and at most 30 seconds for an ordinary event or 45 seconds for a
reported-visible news event. Raw model outputs remain present and annotated so
the filter cannot hide mistakes from audit.

Replaying that contract over all 14 reviewed v3 outputs filtered exactly four
manual rejects and no accepts. This useful result is still not a pass: only 2/10
survivors were clean, five were definite false positives, and three required
unresolved media repairs. The residual errors were visually identifiable but
not reliably transcript-identifiable: production setup around a real utterance,
talking-head policy discussion mislabeled as enacted dialogue, press conferences,
mixed event/debrief intervals, and split-screen label leakage. The supported
architecture is therefore a transcript detector for candidate generation followed
by an exact-span visual gate. No transcript-only prompt may serve as the keep gate.

All three v3 repair candidates were then rendered as exact hashed MP4s and fully
reviewed. The 8.24-second silent meal recut passed; an expanded 6.28-second child
scene passed after the small thrown object was confirmed in flight and the label
was changed from `correct` to `contrast`; and the 16.36-second silent right-panel
news crop passed after isolating the crowding behavior. This raises eventual usable
yield to 5/14 (35.7%), but does not rescue the detector: three of the five usable
examples depended on manually chosen temporal, audio, spatial, or semantic repairs,
and the original source outputs retain their failed/uncertain decisions.

### 07:06 fresh hourly audit

The next source-disjoint window contained 389 raw instructional rows, nine raw
commentary videos, and no witnessed rows. Source-level freshness exclusion removed
149 instructional rows and all nine commentary videos because they had already
been judged or assigned to an audit batch, leaving 240 fresh instructional clips.
Twenty-four were sampled, exported at up to 12 temporal positions, and all 259
frames plus aligned transcript spans were reviewed under instructional v4.

Only 1/24 clips was usable, and only after semantic relabeling: a staged store
security employee explicitly racially profiles a Black shopper. The other 23 were
talking heads, interviews, static text, unrelated B-roll, or technical, animal,
fitness, health, finance, poker, art, and self-improvement content. The apparent
WWYD intervention candidate was a post-event recollection followed by the start of
a different narrated setup, not the intervention itself. Yield was 1/5 for
`violation`, 0/7 for `correct`, and 0/12 for `explanation`, or 4.2% overall.
This interval independently confirms that neither didactic intent nor a social
topic is enough: the exact pixels and aligned dialogue must contain the event.

### 07:46 witnessed retrieval and scheduler audit

The absence of witnessed rows was traced to deterministic scheduler starvation,
not a quiet detector. During the preceding 24 hours every processed source was
from an `instr_*` category: 499 active priority-3 instructional queries remained
above commentary at 2.5, candid/witnessed at 2.0 or lower, and null at 1.2. The
scheduler orders by priority before least-recently-used time, so lower families
cannot run while the priority-3 pool remains active. This supports an alert and
read-only family canaries, but not an unaudited live priority change.

A frozen Dailymotion retrieval shadow then compared six existing witnessed
queries with six paired action→reaction queries. All 240 returned titles were
reviewed. The paired wording helped only the racism/confrontation vein; in most
queries Dailymotion dropped the action noun and returned generic
“caught-on-camera” footage, animals, toys, products, pranks, or unrelated senses
of “line” and “phone.” A non-cherry-picked rank sample retained the top two
unseen results per query, advancing only to avoid duplicate UIDs. Nineteen of 20
artifacts downloaded; all 19 were reviewed at 16 temporal positions, eight were
transcribed, and five ambiguous cases received 32–40-frame dense reviews.

Strict witnessed yield was 0/8 for the existing baseline and 0/12 for paired
action→reaction v1. The closest organic cases still failed: a littering clip
started after the discarded box was already on the ground; a doorbell clip did
not make the alleged theft visually unambiguous without its title; and a
fast-food assault/intervention began reacting before a complete, useful
pre-reaction action could be spliced. Three staged or animated clips are only
instructional reroute candidates, and one narrated police-misconduct video is
only a commentary candidate; none is accepted without its pillar-specific
follow-up. Therefore the paired query rule is frozen as failed and must not be
promoted.

The three instructional reroutes then received that separate follow-up and all
three failed. The animated dog-waste clip could not resolve failure to clean up
without retaining a thought-bubble/checkmark answer overlay. The grocery prank
did not make ownership or the alleged taking action unambiguous without its
title. For the anti-litter PSA, a muted 23.5–33.5-second repair was rendered and
hash-verified; all ten sampled frames were reviewed. A foreground occlusion hid
the critical transition and the can was next visible already in hand, so the
proposed pickup was not actually shown. An earlier input-seek render also failed
duration validation at 9.64 rather than 10.0 seconds and was never imported.
The successful output-seek render was 9.96 seconds with zero audio streams, but
exact media validity did not rescue its semantic failure. Instructional reroute
yield from this search shadow is therefore 0/3, with no corpus mutation.

### 08:20 post-batch hourly audit

The scheduled batch ran on allowed GPU 5 and drained 274 transcription inputs.
Detection completed with 116 instructional routes, five commentary routes, 143
drops, four misses, and zero pending detections; the later transcription backlog
was 65 and the crawler remained healthy. All 268 processed sources in the hour
still came from `instr_*` queries. The successful batch therefore confirms that
the continuing absence of witnessed outputs is scheduler starvation, not a dead
pipeline. No live priority change is authorized by this observation.

The fresh, source-disjoint pool contained 542 instructional clips, five commentary
videos, and zero witnessed clips. The 24 selected instructional clips were escalated
from a three-frame screen to full instructional-v4 review: all 228 temporal frames
and aligned transcript spans were examined. Only 2/24 (8.3%) passed, both animated
but visually resolved social demonstrations: a deer declines a friend's request
for protection from wild dogs, and three friends pool snacks and drinks so everyone
can eat. This again supports medium neutrality. The other 22 failed, with overlapping
counts of 15 lacking any resolved temporal demo, 11 off-topic, six not social norms,
and five explanation/talking-head items. Yield was 1/8 for `correct`, 1/7 for
`violation`, and 0/9 for `explanation`; all came from current instructional queries.

All statements—not just the first detector output—from the five commentary videos
were then reviewed with 20 seconds of transcript context. Sixteen of 25 passed only
after concrete norm relabeling and nine failed. Manual event-plus-behavior
deduplication retained 11 clusters. A compound transport-policy sentence was not
retained because narrower grounded statements separately covered assistance-animal
refusals and short-fare refusals; four descriptions of one MLA/teacher incident
collapsed to the strongest explicit behavior-plus-judgment statement; and two
assessment-preparation complaints from one TAFE incident collapsed together.
Automated clustering was not used because both prior shadow prompts failed exact
source partitions. All 49 judgments passed ledger validation and were imported into
the audit ledger only; no corpus labels, detector prompts, routing, or query priorities
were changed.

### 08:55 witnessed retrieval shadow v2

A second read-only witnessed canary used eight targeted action-plus-confrontation
queries. All 160 returned titles were reviewed, then the top two unseen,
non-title-skipped results per query were frozen without cherry-picking; the
all-seen racist-rant query contributed no item, leaving 14 selected results. One
download failed and the other 13 videos were reviewed at 16 temporal positions.
Four ambiguous organic-looking cases received 168 additional dense frames and
aligned transcripts.

Only one result survived. In `dailymotion__x55wu8f`, a driver faces the cyclist
and calls him a profane insult at 10.949–12.970 seconds; the cyclist's first
explicit targeted objection begins at 14.731 seconds. The proposed 10.500–13.300
second training interval was rendered as a new audiovisual MP4, duration-checked,
hash-verified, sampled again, and fully reviewed. The artifact preserves the
complete insult and excludes the weak-label reaction. Its SHA-256 is
`876f63dc7322b49ae5adb6f3e1a15dab842d37bab09be08e2a2e80bf1bbc095c`.

The other three dense cases failed closed: one recording started after the
alleged vehicle collision and only narrated it; one began during a blurred
incident and had no targeted normative reaction; and one started mid-bakery
dispute without reliably attributable evidence of the title's alleged threat.
The remaining results were mundane CCTV, police B-roll, graphics, games, sports
talk, or news packages. Strict yield was therefore 1/14 selected (7.1%), or
1/13 viewable (7.7%). This supports only a source-disjoint replication of the
driver/cyclist query vein. It does not support activating the v2 query family,
auto-accepting results, or changing production routing.

### 09:08 source-disjoint hourly audit

The 09:07 scheduled pipeline check found 147 pending transcriptions, zero
pending detections, and an oldest item age of one hour. It correctly exited
under the existing 150-item/four-hour threshold without touching a GPU. The
hourly audit still sampled the unreviewed routed pool. Source-level exclusion
removed 159 instructional rows and all five commentary videos already judged
or batch-assigned, leaving 383 fresh instructional clips and no fresh
commentary or witnessed items.

The frozen 24-clip sample was escalated to full instructional-v4 review. All
262 temporal frames and aligned transcript spans were examined. Two clips
passed: a live role-play in which participants use ethnic/religious labels and
make a stereotype-based threatening remark passed after a concrete norm
relabel, and an organic caregiver-child scene passed cleanly because the adult
visibly and verbally reassures a distressed toddler after a perceived accident
and prompts repair. The other 22 failed. Overlapping failures included 16 with
no resolved visual demo, 15 that were not social norms, 15 off-topic items, 12
with visible answer-label leakage, and nine technical/procedural lessons.

Yield was 2/24 (8.3%): 1/5 violation, 1/8 correct, 0/10 explanation, and 0/1
contrast. The accepted live-action role-play and organic scene again support
medium neutrality, while the zero explanation yield and recurrent software,
music, driving, language, reading, fitness, and self-improvement lessons
reinforce the exact-scene gate. All 24 judgments passed the compiler and ledger
validators and were imported only into the append-only audit ledger; production
metadata, detector prompts, routing, and query priorities remain unchanged.

During this escalation, an initial export command pointed at a newly created
non-authoritative audit database rather than `data/visual_audit/audit.db`. That
attempt never touched corpus state or the authoritative ledger. The same 24
frozen item IDs were rerendered against the authoritative database; all 24
per-item frame-manifest hashes matched the reviewed artifacts exactly. The
judgments were then recompiled with the authoritative batch ID and all 24 were
inserted there. The non-authoritative database and report were renamed and are
excluded from official coverage. The corrected report records 290 instructional
v4 items reviewed, with the policy still inactive and not ready for promotion.

### 09:30 threshold-qualified batch and post-batch audit

Because the queue crossed the configured threshold just after the 09:07 cron
check, the unchanged flocked batch pipeline was invoked once at 09:20. Its
normal selector used allowed GPU 5; reserved GPUs 1–4 were untouched. Of 178
transcription inputs, 176 completed and two errored. Detection routed 78 to
instructional, ten to commentary, dropped 86, and missed two; it drained to
zero in two minutes. The crawler remained healthy and had added 23 new
transcription rows by completion. No witnessed route appeared.

The post-batch source-disjoint audit reviewed all 270 frames and aligned spans
for 24 instructional clips. Only 1/24 (4.2%) passed, after relabeling a broad
norm: volunteers visibly accompany, steady, and include autistic children in a
group horse-riding activity. Yield was 1/8 correct, 0/9 explanation, and 0/7
violation. The 23 failures again overlapped heavily: 16 lacked a resolved demo,
14 visibly supplied an answer label, 12 were not social norms, 12 were
off-topic, and nine were technical procedures.

All 24 statements from the ten new commentary videos were reviewed with full
context. Thirteen passed at statement level—two clean and eleven after concrete
norm relabeling—and eleven failed. Manual occurrence deduplication retained nine
clusters and removed four duplicate statements. A broad pattern describing
racialized enforcement during driving, walking, and sitting was deliberately
kept separate from the named Starbucks arrest that was one constituent event;
this follows the earlier anti-overmerge audit rule. All 48 judgments passed the
authoritative ledger validators and were inserted only into
`data/visual_audit/audit.db`. No detector, routing, query-priority, or corpus
metadata change was made.

### 09:41 instructional retrieval shadow v1

A read-only Dailymotion canary tested ten explicit role-play, video-modeling,
WWYD, and social-experiment queries. All 200 returned titles were reviewed. The
top two unseen, non-title-skipped results per query were frozen without
cherry-picking; the autism video-modeling and acted conflict-resolution queries
had no unseen top-20 results, leaving 16 selected sources. Every selected source
was downloaded and reviewed at 16 temporal positions.

Nine plausible or ambiguous sources then received 469 dense frames and complete
WhisperX transcripts. WhisperX incorrectly classified one English salon clip as
Welsh, so that source was rerun with English forced and the new transcript was
reviewed before judgment. Seven candidate demo intervals were subsequently cut
as new audiovisual MP4s, duration-checked, hash-linked to their source media,
and reviewed over 112 additional frames. All seven exact artifacts passed:

- a French dramatization of coded housing discrimination;
- two salon role-plays, one showing poor service and one showing helpful
  clarification of a customer's product request;
- a staged street-harassment and assault PSA;
- a WWYD sex-discrimination hiring scenario;
- a public bullying/extortion experiment; and
- an audio-critical video-chat bullying experiment.

Only four of those seven matched the norm targeted by their retrieval query.
The two salon clips came from the consent query but were customer-service demos,
and the harassment PSA came from the expected/unexpected-behavior query. They
are recorded as reroutes, not query successes. Thus target-matched exact-demo
yield is 4/16 (25.0%), while recoverable exact-demo yield is 7/16 (43.8%). The
other nine sources were stock/legal graphics, technical slides, a job-role
talking head, an unresolved drama montage, an anti-bullying talking head, a fail
compilation, two duplicate conference-panel uploads, or an off-query dating gag
without a stable violation label.

The evidence supports source-disjoint replication only for `workplace
discrimination role play scenario`, `what would you do discrimination
scenario`, and `social experiment bystander intervenes bullying`. It does not
support activating the query family. Query and title remain retrieval hints;
off-target demos must be freshly relabeled from audiovisual evidence, and every
candidate must still pass an exact-clip review. No live query, detector, routing,
priority, corpus label, or corpus media was changed.

### 10:29 instructional retrieval shadow v2

The source-disjoint replication canary tested eight queries: four rewrites of
the strongest v1 veins and four rewrites of failed v1 queries. All 160 returned
titles were reviewed. Selection excluded every UID from the complete v1
enumeration, then froze the top two unseen, title-eligible results per query
with global source-UID deduplication. One query had only one fresh result, so 15
sources were selected. Fourteen downloaded and were reviewed at 16 temporal
positions; the failed download remained in the denominator.

Three ambiguous or potentially usable sources received 272 additional dense
frames. Full transcripts were reviewed for the 13-minute ASMR job-interview
role-play and the 7-minute prank compilation. The ASMR source remained a
single-person POV procedural simulation with no discrimination or violated
tacit norm. The toy-kitchen commercial showed children operating the product
separately, not sharing or taking turns. The prank compilation contained one
real enacted sexual-harassment interaction, but it was off-target for the
bullying/bystander query.

The only two target-matched results were separate UIDs for the same WWYD
gender-discrimination episode already audited in v1. The two current uploads
were frame-for-frame near-identical to each other (mean 16-frame dHash Hamming
distance 0.4375) and shared the underlying episode with v1 `x2rcgal`; they do
not count as independent replication. Independent target yield was therefore
0/15, not 2/15.

The off-target prank was audited rather than discarded by title. Dense frames
and 255 aligned transcript segments localized a sexual proposition at
33.9–36.7 seconds and repeated refusal/escalation through 53.6 seconds. The
first action render failed because a viewer-comment overlay repeated the
prompt. A deterministic `1280:560:0:0` top crop removed that overlay while
retaining both participants and the complete action audio. The new 2.802-second
artifact was duration-checked, audio-stream-checked, hash-linked, and manually
reviewed; SHA-256 is
`e13aa15db9ba360b0d22ff2ee79056110dfcbd2373537abf1dea17a1b882c885`.
The longer refusal/escalation render is retained only as private weak-label
evidence and never as training media.

This yields one unique off-target research candidate (1/15) but zero evidence
for any v2 query. No v2 query is promoted. Future replication audits must be
episode-disjoint, not merely UID-disjoint; reroutes remain separate from query
precision; and target-response harassment must be tested as its own read-only
vein before any witnessed or instructional policy expansion. No live query,
detector, routing, priority, corpus label, or corpus media was changed.

### 10:34 episode-level retrieval dedup calibration

Because UID-level exclusion admitted duplicate uploads of an already audited
episode, a new audit-only candidate generator compared all 30 rendered v1/v2
search-shadow sources. It uses a strict duration match plus temporal perceptual
frame hashes; lightly edited episodes additionally require title-token overlap.
It never deletes or rejects anything automatically.

All 435 source pairs were scored. Exactly four pairs were flagged and every
flag was manually reviewed against both 16-frame sheets. All four were true:
the two v2 WWYD encodes were near-identical; each was also a light edit of the
same v1 WWYD episode; and the two v1 Aspen Ideas Festival uploads were
near-identical encodes of one panel. There were no false candidate pairs in
this calibration. This supports using the detector only to require manual
episode-identity review before a source is counted as independent replication.
It does not establish recall or transfer, and does not authorize automatic
deletion, rejection, or corpus mutation.

### 11:04 direct-target-response retrieval shadow v1

A read-only canary tested eight direct-target-objection query formulations.
All 160 returned titles were reviewed, and the top two unseen, episode-disjoint
results per query were frozen without cherry-picking. Fifteen of 16 selected
sources downloaded and received 240 sparse frames; the failed download remained
in the denominator. Eight ambiguous sources then received 672 dense frames.
Seven complete transcripts were reviewed. The eighth source had no audio stream
and failed closed rather than being treated as a silent objection example.

The proposed direct-target-response expansion failed: zero of 16 selected
sources supplied an independent, organic violation that both began on camera
and completed before the target's first causal objection. One source did pass
the existing witnessed bystander rule. In `dailymotion__x7srler`, a man visibly
collapsed and dragged an unhoused person's tent from 0.0–15.95 seconds; the
recorder's first objection began at 16.19 seconds, and the owner later confirmed
that his property had been destroyed. The action and private evidence were
rendered separately, duration/audio/hash checked, and reviewed over 32 exact
frames. Their SHA-256 values are respectively
`3440ff599a7aacddca1dea50e376fb8e9b8953f4e10ced248d0055b9b2f863dc`
and
`ccf6244e43816b222b92082b7652d9b0df6b72dba48305a22111980e19f8c5a2`.
This replicates the current bystander logic; it does not validate target
objection as a new keep signal.

The failures are informative. One selfie recording began after the alleged
following and attempted car entry, so it had evidence but no on-camera action.
The burkini scene's target objected before the fake officer's first resolved
discriminatory demand. A silent physical gag had neither an audio stream nor a
grounded social violation. A staged catcalling compilation did contain direct
refusals, but its produced setup makes it instructional, not witnessed. The
remaining sources were interviews, screenshots, fitness, UFO, geopolitical
news, or unrelated pranks.

Five sources nevertheless yielded seven exact instructional demos: real NYC
street-catcalling footage; a burkini-discrimination social experiment; a
catcalling/refusal experiment; three distinct harassment/intervention scenarios
from a Filipino TV package; and a positive blindfolded-Muslim trust/hug scene.
The first exact pass exposed two loose boundaries—an interview shot before the
NYC demo and a definition interval that ran into the next scene. Both were
rejected, rerendered, and passed only on their second 17-frame audit. Across the
first and repaired passes, 186 exact frames were reviewed and no interval
remained unresolved.

This audit also separates two instructional products that the previous binary
gate conflated:

- A **clean demo** has an enacted social occurrence and no external answer
  label overlapping that exact interval. In-character dialogue and subtitles
  are part of the behavior, not answer leakage.
- A **labeled demo** has a genuine enacted occurrence but overlapping external
  narration or an explicit answer label. It is usable as clear weak supervision,
  but is excluded from leakage-sensitive evaluation and audio-enabled
  label-prediction benchmarks.

The three TV scenarios are labeled demos because narration overlaps them. The
trust experiment's opening text is a scenario prompt, not an answer judgment;
the later prescriptive advocacy cards were excluded. This preserves the user's
medium-neutral requirement—staged scenes, documentaries, TV packages, and
positive demonstrations are all admissible—while still tracking shortcut risk.
A fixed demo-per-video cap is not supported: the TV source contained three
distinct enacted occurrences, each separately bounded and reviewed. Multiple
near-duplicate extractions of one occurrence remain disallowed.

The episode scan expanded to all 77 rendered search-shadow sources and all
2,926 pairs. It returned only the same four previously reviewed duplicate
instructional pairs; no target-response source formed a candidate pair. All
four were revalidated through unchanged immutable sheet/media hashes and
metrics. No query, detector, router, priority, corpus label, or corpus media was
changed.

At 11:04 the crawler remained healthy. The queue held 295 pending
transcriptions, zero pending detections, and an oldest age of 1.74 hours. That
qualified for the normal threshold but was left to the scheduled 11:07 flocked
batch; the monitor did not take a GPU, and reserved GPUs 1–4 were untouched.

### 11:19 scheduled-batch audit

The scheduled pipeline completed at 11:18 on allowed GPU 5. It selected 303
transcription rows, completed 298, and recorded five errors. Detection routed
124 videos to instructional, eight to commentary, dropped 162, and missed four.
The crawler remained healthy, `pending_detect` returned to zero, and GPUs 1–4
were untouched.

The source-disjoint post-batch sample contained 24 instructional clips and all
seven new commentary sources; the interval produced no witnessed route. The
instructional review did not stop at the sampler's 72 frames. Every clip was
escalated across its complete exact span to 12–48 uniformly spaced frames, for
363 additional frames, and every frame page was manually reviewed. Eight of 24
clips contained some concrete visual demonstration, but three were CPR,
driving, or dog-training procedures rather than social norms. Only five of 24
(20.8%) depicted norm-relevant social conduct, and only two of 24 (8.3%)
supported their current weak labels.

One clean pass was an English-learning role-play in which spouses enact a
forgotten-anniversary conflict. The second pass used produced hospital B-roll:
a worker and kosher menu concretely demonstrated religious dietary
accommodation. Because external narration overlaps that behavior, it is a
training-only labeled demo, not leakage-clean evaluation media. This is direct
evidence for keeping the medium-neutral rule: animation, role-play, B-roll,
documentary, and live action may all pass, but only a concrete exact-span demo
can supply the visual weak label.

Three other clips contained potentially useful social action but failed their
current labels. A 121-second children's clip joined several unrelated prank,
property, and balloon episodes under one sharing violation. A satirical
environmental interview was labeled as correct caring even though the visible
device signaled indifference. A restaurant demonstration explicitly taught
that a reduced tip is appropriate after bad service, while the stored polarity
called the small tip a violation. These cases support occurrence-level temporal
localization and an independent norm/polarity check; they do not support a
duration-only filter or automatic relabeling.

All 26 statements from the seven commentary sources were then reviewed against
their timestamped context. Six (23.1%) passed with the stored norm, nine (34.6%)
were recoverable only after a behavior-specific relabel, and eleven (42.3%)
failed. The failures included five song-lyric fragments, an animal reaction, a
hypothetical comedy chant, victim testimony without an explicit normative
stance, and descriptive social context without judged conduct. Six overlapping
fragments in one parliamentary-harassment report also showed why event-level
collapse is necessary. Generic labels such as `respect`, `safety`,
`sensitivity`, and `accountability` cannot be final labels unless the concrete
judged behavior is separately normalized and grounded.

The complete review is frozen in
`audit_runs/20260721_111942_hourly/manual_summary.json`. This cohort failed the
promotion gate. No prompt, detector, router, query priority, corpus label, or
corpus media was changed.

### 11:46 instructional transcript-shadow v4 audit

The tighter transcript detector was then run in shadow mode on all 24 frozen
instructional sources from the 11:19 cohort. It selected five sources and emitted
16 proposed exact spans. Every proposed span was reviewed, including the five
that failed deterministic checks: 15 were rendered as exact proxy clips, all 359
uniformly sampled frames were inspected, and the nonpositive interval failed
closed. Exact transcript context was checked for all 16.

Only one output passed unchanged: a staged household exchange in which one
speaker politely reminds others about shared chores and the recipients explicitly
thank that speaker. Fifteen outputs failed. Most importantly, only 1/11 (9.1%)
outputs that passed the deterministic transcript contract survived the manual
visual and semantic gate. The ten false accepts included hand washing, eating,
litter disposal, and silverware use that were described but never shown; a
lawsuit and waste-disposal practice that were only reported; a podium speech; a
wrongly labeled party reminder; and two historically specific 1970 US military
gender-etiquette scenes emitted as universal present-day rules.

The shadow also missed known valid material: it rejected the previously reviewed
hospital dietary-accommodation B-roll, and it truncated a clean
forgotten-anniversary role-play before the anniversary was disclosed. This is a
precision and recall failure, so no transcript prompt or deterministic contract
is eligible for promotion. In particular, a transcript model's self-reported
`evidence_kind` is not visual evidence.

The required architecture is now explicit: transcript extraction may propose a
candidate interval, but an exact-span visual verifier must establish that the
behavior is actually shown or performed in a grounded social exchange, that the
span contains one resolved occurrence, and that norm, polarity, and any temporal
or cultural scope match that occurrence. The server currently has no vision model
cached or serving, and no approved closed-VLM API credential is exposed. The
existing 70B detector is text-only. Therefore automatic acceptance remains off;
the next model calibration cannot run until an approved endpoint exists, and its
complete output diff must again receive 100% manual review.

The immutable result is in
`audit_runs/20260721_111942_hourly/instructional_shadow_v4/manual_summary.json`
and the row-level decisions are in
`audit_runs/20260721_111942_hourly/instructional_shadow_v4/rendered_outputs/manual_review.tsv`.

### 11:53 commentary same-event shadow v3 audit

The 26 already frozen, full-context commentary statements were next run through
a stricter text shadow. It requires actor, concrete behavior, affected target or
shared context, and an explicit stance to refer to the same occurrence; behavior
and stance evidence must be verbatim; generic proposed norms require an explicit
relabel. Every one of the 26 outputs was compared against the prior manual
statement-level judgment.

Twelve raw model outputs violated the deterministic contract and failed closed.
The common error was internally inconsistent output—`accept` paired with an
unsupported proposed norm—or invented/paraphrased evidence rather than a
verbatim quote. This contract caught the cohort's dangerous false accept: a
reported attack/kicking claim for which the surrounding text did not explicitly
judge that conduct.

After fail-closed enforcement, two statements were retained and both were true
accepts: the Urdu news report explicitly grounded a teacher beating a child for
not memorizing a lesson and the child's loss of consciousness. Effective
precision was 2/2 with zero false accepts, but recall was only 2/15 (13.3%) of
manually usable statements. Thirteen usable statements were lost, especially
where a vague proposed norm needed relabeling or the linked behavior and stance
occurred in separate context.

This is promising only as a precision-tier canary. It is not promoted: the raw
contract failure rate is 46.2%, the cohort covers just seven sources, and the
zero-false-accept result needs source-disjoint replication. The frozen summary is
`audit_runs/20260721_111942_hourly/commentary_shadow_v3/manual_summary.json`; all
26 row-level comparisons are in the sibling `manual_review.tsv`.

The unchanged prompt and contract were immediately replicated on a fully
source-disjoint earlier cohort: 25 statements from five different videos with 16
manually usable labels. All 25 outputs were reviewed. Fifteen failed the contract,
and the remaining ten were rejects, so the effective gate retained zero items.
The contract did catch all four raw false accepts, but it also lost all 16 usable
statements. A zero-output gate does not replicate the first cohort's positive
precision and is not useful weak supervision. Shadow v3 therefore fails rather
than passes. Its source-disjoint audit is frozen at
`audit_runs/commentary_shadow_v3_replication_082006/manual_summary.json`.

### 13:19 fresh hourly instructional audit

The next source-disjoint two-hour window contained 145 instructional demo rows
and no witnessed or commentary outputs. All 24 selected instructional clips were
escalated from the sparse screen to dense temporal review. The exact transfer was
hash-verified; 569 frames across 49 contact-sheet pages and all 24 aligned
transcript spans were manually inspected. The authoritative audit ledger now
contains 24 judgments for batch `instr_20260721T202103Z_e3b7b046`.

Only 2/24 (8.3%) current labels passed. Both were concrete animated social
exchanges: introducing a friend and giving a person specific directions. This is
positive evidence for the medium-neutral rule—animation and visible
in-character text are valid—and not evidence for accepting generic educational
animation. Ten clips demonstrated some temporal action, but only three depicted
social conduct, and one of those was a multi-couple argument montage whose
stored `conflict resolution` label was unsupported. Technical demonstrations
(Java, device care, hygiene, animal handling, basketball, and dance) therefore
remain explicit rejects even when their pixels clearly show an action.

Explanation polarity again failed completely (0/12). Correct polarity passed
2/10 and violation passed 0/2. The most common overlapping failures were
off-topic (14), not a social norm (14), no visual demo (14), technical or
procedural content (7), and talking head or lecture (7). A particularly useful
boundary failure said “have a peer demonstrate aspects of a game,” but six of
seven frames showed only the instructor and the final frame merely established
seated participants. A transcript promise of a future demo cannot satisfy the
visual gate.

This cohort supports four strict shadow rules: require an actual visible social
occurrence in the exact selected interval; require current norm and polarity to
match that occurrence; exclude technical/medical/animal/physical-skill actions
unless the demonstrated action is itself interpersonal or concerns a shared
social context; and treat explanation as label context rather than a demo
polarity. It does not support a production change. The immutable summary and
row-level decisions are in `audit_runs/20260721_131910_hourly/manual_summary.json`
and `dense_audit_v2/manual_review.tsv`.

### 13:35 source-disjoint witnessed and commentary canaries

Before selecting this cohort, the audit exporter was found to violate its own
freshness claim: it excluded only an exact prior item, not every item from the
same source video, and a witnessed export incorrectly queried the instructional
judgment table. The exporter now excludes a pillar/source UID after any prior
judgment or batch assignment and uses the correct pillar-specific judgment
table. Commentary sampling now applies the same source-level rule. Three new
regression cases cover instructional prior assignment, witnessed prior judgment,
and commentary prior assignment; the complete suite passes 44/44.

The corrected witnessed canary contained 16 clips from 16 never-assigned source
videos. Every aligned span and all 256 initial frames were reviewed. Two
organic-looking cases were expanded to exact MP4 review at four frames per
second, adding 140 frames. Both failed closed: paparazzi footage contained a
grounded “don't push me” objection but never resolved who pushed whom, while a
roadside vehicle clip contained repeated “shut the door” commands but the
vehicle pillar occluded the triggering reach/opening action. Final decisions
were zero strict witnessed accepts, one scripted instructional reroute, and 15
rejects. The reroute was a Seinfeld scene that visibly demonstrates double
dipping and a targeted objection; it is useful scripted instructional evidence,
not organic witnessed supervision.

The source-disjoint commentary canary contained 24 statements from 24 videos,
each with 20 seconds of transcript context on both sides. All 24 were reviewed:
three retained the stored norm, ten contained a usable same-event behavior and
stance only after semantic relabeling, and 11 were rejected. Event-level yield
was therefore 13/24 (54.2%), but clean current-label precision was only 3/24
(12.5%); 76.9% of usable statements required relabeling. Bare insults could not
label themselves, event descriptions without a stance failed, and missing
actors could not be supplied by titles. The accepted cases included grounded
theft criticism, police-force condemnation, road-blocking rules, family
boundaries, vandalism criticism, unlawful detention/due-process advocacy,
anti-violence commentary, racial-exclusion correction, and consent before
workplace touching.

These results strengthen different rules for the two pillars. Witnessed data
requires separately identified behavior and reaction speakers, visible causal
order, organic authenticity, and a complete pre-reaction splice. Commentary can
use separate nearby spans, but both verbatim quotes must concern the same
occurrence; semantic relabeling is a mandatory audited repair rather than an
automatic normalization step. Neither cohort supports a production change. The
complete row-level evidence and hashes are frozen in
`audit_runs/20260721_1335_source_disjoint/manual_summary.json`.

### 13:57 atomic social-scope shadow and commentary v3 comparison

The holistic `is_social_norm` question was removed from a new shadow runner. A
model or reviewer now supplies four enumerable premises: actor kind, behavior
kind, affected-context kind, and expectation kind. Code deterministically derives
`social_actor_grounded`, `concrete_behavior`,
`target_or_shared_context_grounded`, and `is_social_norm`; a model cannot answer
those composites independently. Explicit technical procedures, accidents,
nonhuman behavior, private self-improvement, technical correctness, legal-only
rules, safety-only rules, and personal preferences derive `no`. Missing actors,
actions, targets, or unresolved premise types derive `uncertain` and fail closed.

Every item in three frozen source-disjoint/manual cohorts was re-annotated under
the atomic contract: 24 instructional, 16 witnessed, and 24 commentary items
(64/64 total). Every previously usable item retained derived scope `yes`; no item
was promoted. Five already-rejected holistic `yes` judgments became `uncertain`
because the responsible actor or concrete occurrence was absent. The contract
therefore passed its human shadow audit, but remains non-production until the
intended closed-model outputs can also be reviewed 100%. No configured
unattended OpenAI credential or available GPT-5.6-sol endpoint existed on sk3
during this run. The
row-level premises and immutable summary are in
`audit_runs/20260721_atomic_scope_shadow_v1/`.

The corrected local Llama-3.3-70B commentary v3 rerun was also compared manually
for all 24 items. It accepted 20; only 12 were manually usable, for 60% accept
precision. Fourteen outputs failed the deterministic contract. Even among the
six contract-valid accepts, two were semantic false positives (an inferred actor
from passive voice and a legal-only railway offense), giving 4/6 precision. Exact
manual decision agreement was 6/24. The v3 rule is rejected and must not be
promoted. Complete row-level dispositions are frozen beside
`commentary_shadow_v3_manual_summary.json`.

### 14:10 fresh hourly instructional and commentary audit

The next source-disjoint window contained 476 instructional demo rows, ten
commentary videos, and no witnessed output. All 24 selected instructional spans
were reviewed over 438 chronological frames. Four ambiguous cases were then
downloaded as exact MP4s and inspected at two or four frames per second. Only
one clip survived (1/24, 4.2%): an animated, in-character invitation and
agreement to help find books, which visibly grounds correct cooperation. None
of ten explanation clips or four violation clips survived. Eighteen clips had
no temporally grounded visual demo, 16 were not social norms, 16 were off topic,
and eight were technical procedures; these reasons overlap.

The exact escalations prevented three tempting false accepts. Anti-harassment
metadata pointed to an expert interview with assorted stock encounters, but no
complete intervention was shown. LEGO therapy news footage showed people in a
group without demonstrating a specific labeled social norm. A film montage had
audio containing “Don't harass me,” but the ten-second image sequence jumped
among unrelated scenes and never isolated a visible harassment act; its stored
“Respect authority” norm was also wrong. These remain rejects rather than repair
candidates.

All 23 commentary statements from the ten videos were separately reviewed with
20 seconds of transcript on each side. Four survived (17.4%): one clean honesty
case and three specific behavior/stance pairs after norm relabeling. Nineteen
failed, predominantly because a concrete behavior lacked explicit criticism or
a criticism lacked a concrete behavior. Multiple detector spans for the same
event were collapsed manually to the strongest grounded member. All 47
judgments were imported into the append-only shadow ledger; production routing
and corpus metadata were not changed. The immutable summary is
`audit_runs/20260721_141041_hourly/manual_summary.json`.

## Hourly monitoring protocol

Run:

```bash
python3 scripts/fetch_visual_audit.py \
  --since-hours 1.5 --instructional 24 --witnessed 16 --commentary 16
```

The command runs the sampler on sk3, copies only a small frame bundle, and
creates blank visual/commentary review TSVs. Every visual row has frames at 15%,
50%, and 85% of the extracted clip. The sampler excludes an entire source video
once any of its items has a prior judgment or prior batch assignment; this
prevents overlapping 90-minute windows and multi-demo videos from masquerading
as fresh evidence. Report:

- pool and sample counts by pillar;
- instructional demo rate overall, by polarity, category, and query family;
- witnessed visible-scene rate and strict organic weak-label rate;
- commentary statement precision;
- top rejection reasons and the UIDs for every defect;
- current backlog and new-output counts.

Repair metrics are reported separately from source precision. A source item may
not inherit an accepted disposition merely because one proposed transform was
rendered; the accepted unit is the exact hashed repaired artifact.

Alert conditions for the next collection interval:

- instructional demo precision below 50% overall or below 20% for any stratum
  with at least ten reviewed clips;
- witnessed strict precision below 70%;
- any `comm_*` or `instr_*` row routed to witnessed;
- off-topic rate above 10%;
- no output from an intended pillar for two consecutive intervals;
- `pending_detect` or `pending_transcribe` increasing for two consecutive runs.

### 14:30 witnessed recovery and 15:00 commentary-video recovery

A further source-disjoint witnessed canary reviewed 24 source videos over 574
frames, then reviewed all nine plausible recovery candidates over 234 denser
frames and exact MP4s where needed. No stored witnessed item passed unchanged.
Two organic sources did yield strict repaired training clips after the reaction
signal was removed: a road-rage approach followed by an on-scene protective
intervention, and a loud hotel party followed by a worker's correction. Four
scripted sources were routed only to independent instructional review; 18 were
rejected. The result replaces the misleading shorthand “0/64 means the pillar
is empty” with the narrower conclusion that the current stored labels and
boundaries are poor while some source videos remain recoverable.

The witnessed v2 shadow contract now derives its decision from atomic fields:
actor and behavior kind, affected context, expectation kind, observed action,
reaction role, reaction grounding and content, temporal order, label relation,
boundary basis, and authenticity. A target's own distress cannot certify a
witnessed label. A bystander, authority/host, or organic audience must object,
correct, or intervene after the action. Nearby reactions and label repairs
produce `recover_strict`, never acceptance of the stored item. Frame grids
cannot certify a splice; the exact reaction-free artifact must be rendered,
hashed, and reviewed.

Commentary previously had no saved visual clips because `save_discussion` was a
text-only sink, not because all 52,539 statements came from talking heads. Only
2/89 latest strict accepted commentary sources had retained raw video; both
contained recoverable events, but that retained-raw cell is strongly biased and
is not a yield estimate. A deterministic source-disjoint redownload canary then
reviewed 12 strict text accepts with no retained raw over 480 frames and 13
successive exact artifacts. Four sources (33.3%) produced a clean commentary
visual clip and one scripted source (8.3%) produced an instructional-review
candidate. Seven remained text only. The combined any-demo rate was 5/12
(41.7%), conditional on upstream strict text acceptance; it must not be applied
to the full corpus before source/event deduplication and a larger canary.

The commentary visual contract requires the behavior itself on camera, exact
same-event actor/target/context grounding, a clear visual or audiovisual demo,
and an exported artifact with no normative answer signal. Actual event footage
inside a news wrapper is represented as `organic_news_footage`; news narration
or generic B-roll is still text only. If narration/captions overlap the event,
the item can pass only after both audio and visible label regions are removed
(`mute_and_crop`) and the exact output is manually re-audited. Crop or mute
alone cannot repair overlapping leakage. Scripted occurrences go through the
instructional gate rather than becoming commentary visuals.

The frozen evidence is in
`audit_runs/20260721_1430_witnessed_dense/`,
`audit_runs/20260721_1430_witnessed_recovery/`, and
`audit_runs/20260721_1500_commentary_visual_recovery/`. No production corpus or
routing metadata was changed.

### 15:33 commentary source-retention correction and backfill

The apparent zero-video commentary corpus was a real storage defect, not an
inventory mistake. `finish_detection` wrote a text-only discussion JSON and then
purged the raw media. At correction time, 16,825 discussion JSON records existed,
16,823 had a source URL, but only 425 unique commentary sources still had raw
video. Those 425 sources were adopted into stable storage and validated with
duration, stream, and SHA-256 checks. Future commentary routing now moves the
source into `data/discussion_video/` before purging expendable raw sidecars and
records `source_video` in new discussion JSON.

A deterministic source-disjoint download canary restored 12 additional files:
ten Dailymotion and two Rumble. All 12 were manually reviewed over 12 overview
and 12 quote-window frames. Every file contained temporally varying, source-
consistent media; none was a black/error placeholder or mismatched download.
This is a storage-integrity pass only, not a visual-label acceptance. The review
is frozen in `audit_runs/20260721_1540_commentary_video_backfill_canary/`.

Platform canaries exposed two independent access failures. Eight of eight
YouTube attempts hit the sign-in/anti-bot gate, and 12 of 12 Reddit attempts
required account authentication. Dailymotion and Rumble passed, so a locked,
resumable, low-priority backfill for those two sources was launched rather than
blocking the recoverable majority. Reddit remains pending a sanctioned
credential path. Every attempt and every retained artifact is recorded in
`data/discussion_video/backfill.jsonl`; the ledger includes source, method,
duration, stream counts, byte size, and SHA-256 for successful artifacts.

This was an explicitly requested source-storage correction, not a weak-label
rule deployment. Commentary text acceptance and visual-event acceptance remain
audit-gated and separate.

### 2026-07-22 three-host YouTube source recovery

Three user-provided, distinct YouTube cookie exports were installed as mode-0600
secrets, one each on sk1, sk2, and sk3. Each host uses an isolated yt-dlp
2026.07.04 environment with the EJS helper, Deno 2.9.3, and ffmpeg/ffprobe; no
training environment or GPU process was changed. An actual audio-plus-video
canary succeeded with every credential, and each canary was manually reviewed
over 12 temporally distributed frames against its source title.

A frozen manifest contains 1,128 YouTube commentary sources that were missing
stable video. Its SHA-256 is
`8670ed9d5f6b70eb42d795d1ae088b93722922c703c8e0d2b09c6e21968984a3`.
Deterministic hashing assigns 67 records to sk1, 559 to sk2, and 502 to sk3.
The workers are low-priority, single-process, node-local jobs with independent
locks and ledgers, a 720p/30-minute cap, ffprobe stream and duration validation,
SHA-256 provenance, atomic staged moves, ten seconds between sources, an
authentication circuit breaker, and disk/byte safety limits. sk1 is capped at
40 GB and must retain 320 GB free; sk2 and sk3 are each capped at 1 TB and must
retain 5 TB and 10 TB free respectively.

The first three bulk outputs from every host were manually audited using 108
temporally distributed frames. All 9/9 files matched their discussion-record
titles and were real, varying, audio-plus-video media. The frozen review is in
`audit_runs/20260722_distributed_youtube_launch/`. This approves source
retention only: no commentary event or visual weak label is promoted until a
separate event-localized audit passes. sk1 and sk2 outputs remain node-local
until a safe authenticated consolidation path is available.

### 2026-07-23 hourly monitor and commentary event-localization bottleneck

An observational monitor now runs on sk3 at minute 42 of every hour. It appends
inventory, process, ledger, authentication, and disk-health snapshots to
`data/monitor/hourly_collection/` and queues four previously unqueued retained
videos for manual review. It cannot restart jobs, download media, change labels,
or promote candidates. Its first snapshot had no alerts; all four queued source
videos were manually reviewed over 48 frames and matched their titles. Evidence
is frozen in `audit_runs/20260723_hourly_collection_monitor_first_run/`.

The next event-localization audit then reviewed the complete set of 13
text-approved commentary sources that had retained video on sk3 and had never
received a prior commentary visual audit. Every source received 16 overview and
24 transcript-centered frames (520 total); four possible survivors received
dense exact-artifact review, including two additional bus sanitization
iterations. One organic bus-entry action passed the commentary contract after a
muted, cropped exact cut; one scripted denial scene was routed to instructional
review. Nine remained text only, one low-resolution CCTV source remained
uncertain, and one upstream text label was rejected.

The audit also found one clear muted product-interface demonstration, but the
current contract has no mute-only product-demo transform. It remains unpromoted:
one case is not sufficient evidence to extend the rule. No production rule or
label was changed. Complete judgments, hashes, sheets, and exact artifacts are
in `audit_runs/20260723_commentary_event_localization_bottleneck/`.

### 2026-07-24 source-disjoint three-pillar replication

Three fresh frozen holdouts were completed without deleting, quarantining, or
relabeling corpus items.

The instructional cohort contains 120 source-disjoint clips and 1,440 temporal
frames. All ambiguous items received exact follow-up. Thirty clips satisfied the
manual instructional contract: any visual medium is acceptable, but a concrete
social behavior demo must actually occur. The unchanged GLM-4.6V-Flash plus
Qwen3-14B rule selected 13 clips and, after an append-only correction recognizing
a static illustrated demo, had 10 true positives. Precision was 76.9% and recall
33.3%, below the 90% promotion threshold. False positives included hallucinated
bodily-noise behavior, an instructor with a passive partner but no enacted
exchange, and a street cleaner incorrectly treated as proof of littering.

The commentary cohort contains 50 unique targets represented by 202 six-second
windows. All 50 were manually adjudicated. Thirteen had an exact usable event
window, yet the frozen rule selected zero. A broader visual-recovery rule found
two valid source-recut candidates among 22 recoverable sources (100% precision
on two selections, 9.1% recall). It is retained only as a manual source-mining
queue.

The witnessed cohort excludes 1,291 previously reviewed UIDs and contains 80
fresh sources. Review was label-blind over 960 frames, followed by dense
two-second motion sheets, audio transcripts, and revealed causality,
authenticity, role, and cut-boundary adjudication. One clip passed the strict
contract; one repost remained provenance-uncertain and failed closed. The other
78 failed. GLM-4.6V-Flash and Qwen3-VL-8B completed all 80 atomic audits using
one-frame-per-second video proxies and aligned transcripts. Invalid or reversed
model-proposed temporal bounds were preserved in raw output and normalized only
to unknown, which cannot create an acceptance. The strict two-model
intersection selected two clips, both manual rejects, and missed the sole
positive. The looser review intersection recovered the positive among eight
selections (12.5% precision). Every one of the 21 clips selected by either model
under any tested band was already manually reviewed and is enumerated in
`audit_runs/20260724_witnessed_atomic_holdout_v1/vlm_strict_replication_report.json`.

No tested semantic rule passed its promotion gate. The outputs remain
non-destructive shadow scores and salvage queues. The full repository regression
suite passed 277 tests after the replication tooling and fail-closed validators
were added.

### 2026-08-06 commentary text-label independent re-audit

All 30 rows in the fresh commentary text cohort were independently adjudicated
again against the immutable first-pass file and its SHA-256. This second pass
asks a stricter downstream question: whether the transcript grounds a particular
occurred actor-action-target event well enough to seed label-directed visual
search. It does not treat a generic definition, aggregate discussion, ambiguous
accusation, bare evaluative insult, or explicitly contested norm as a clean
positive event label.

Eleven of the first pass's 16 accepts survived (68.75% retention; 11/30 of the
complete cohort). Five were downgraded: an ambiguous property accusation, a
costume experiment with directly conflicting judgments, a bare bad-driving
insult with no named maneuver, a generic harassment definition, and aggregate
looting examples without one event anchor. Nothing was deleted. The harassment
definition remains an instructional-demo retrieval lead; six ambiguous and one
contested row remain explicit exploratory queues. The append-only decisions,
hash lineage, validation report, and nine unit tests are under
`audit_runs/20260806_commentary_hierarchical_full_source_v2/fresh_text_cohort_1/reaudit_v2/`.

### 2026-08-06 commentary occurred-event historical transfer

A second, label-blind transfer audit used 60 source-disjoint historical
commentary rows (40 prior accepts and 20 prior rejects), excluding every source
from the 30-row V3 design cohort. Source manifests, prior results, transcript
contexts, blind packets, and manual gold are hash-linked. All 60 transcript
packets were adjudicated before any prior decision or label was revealed.

Twenty-five of 60 transcripts grounded a strict occurred actor-action-target
event; nine remained ambiguous exploration leads, three were generic
instructional-demo retrieval leads, and 23 had no eligible route. After the
blind gold was frozen, the historical commentary gate scored 22 TP, 18 FP, 3
FN, and 17 TN: 55.0% precision (95% Wilson 39.8–69.3%) and 88.0% recall (95%
Wilson 70.0–95.8%). It is therefore useful as a high-recall retrieval gate but
not as a strict event label or visual keep rule.

The 18 false positives divide cleanly into eight aggregate/generic/
hypothetical/fictional rows, five ambiguous action/actor/stance or invented
label rows, and five abstract institutional/political/product-performance rows
without a depictable social episode. The three misses were recoverable only by
full-context relabeling or a clarified scope rule. Codification alone is not an
exclusion: directly observable conduct between people or affecting people in a
shared space can qualify even when law also governs it. Abstract policy,
geopolitics, service performance, and product quality still fail without a
depictable actor-action-target event.

The implemented V3 contract now asks occurred-event scope, action specificity,
actor/target binding, and stance consistency separately; generic definitions
route to instructional retrieval and contested/aggregate cases remain explicit
exploration queues. This transfer audit evaluates the old gate against the new
manual gold, not a V3 model. V3 remains shadow-only until every model output is
audited on this set and the separately preregistered fresh confirmation cohort.
