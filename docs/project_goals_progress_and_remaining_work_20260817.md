# Social-Norm Video Corpus: Goals, Work Completed, Current State, and Remaining Work

**Status date:** 2026-08-17  
**Repository:** `social-norms` / live deployment name `norm-scraper`  
**Primary live corpus host:** `sk3` (`skampere3.stanford.edu`)  
**Purpose of this document:** durable research and engineering handoff notes

---

## 1. Executive summary

The project is building a large, weakly supervised multimodal corpus of social
norm events. The intended training examples are short video intervals that show
a concrete social behavior, paired with evidence that the behavior is socially
expected, disapproved of, corrected, praised, or otherwise norm-relevant. The
evidence can come from three different collection pillars:

1. **Witnessed:** an on-scene response to a behavior, such as an objection,
   correction, intervention, protective response, or other contemporaneous
   reaction.
2. **Instructional:** an explicit statement of a social rule paired with a
   demonstration of the behavior. Any visual format is allowed—live action,
   scripted role-play, animation, puppets, games, social experiments, or other
   representational media—provided a concrete aligned demonstration actually
   occurs.
3. **Commentary:** a narrator, commentator, participant, or observer expresses
   a normative stance toward a concrete event. Commentary is always potentially
   useful as text supervision. It becomes visual supervision only if the event
   itself is also present and localizable in the retained source video.

The corpus also contains several kinds of negative controls. These are important
for both evaluation and later discriminative training.

The collection system is large and operational. The latest frozen physical
inventory, dated 2026-08-11, contains **121,461 retained video files**:

| Corpus component | Retained media |
| --- | ---: |
| Instructional | 52,299 |
| Witnessed | 22,204 |
| Commentary source video | 17,783 |
| Negative controls | 29,175 |
| **Total** | **121,461** |

The label ledger is larger than the media ledger because a video may contain
multiple demonstrations, reactions, or commentary statements. The same frozen
snapshot contains **156,774 weak-label rows**:

| Pillar | Weak-label rows |
| --- | ---: |
| Instructional | 64,888 |
| Witnessed | 32,955 |
| Commentary | 58,931 |
| **Total** | **156,774** |

The central quality finding is that collection scale and weak-label precision
are separate problems. We have successfully accumulated a large library, but
many first-pass labels do not yet define clean visual training examples. The
dominant defects are:

- a relevant behavior is discussed but not shown;
- a scene is visible but does not match the assigned norm or polarity;
- instructional clips contain explanation, slides, or topical B-roll instead
  of a connected demonstration;
- commentary text is valid but the event footage is absent, badly localized,
  or contaminated by label-bearing narration or graphics;
- witnessed clips contain reactions to accidents, safety events, animals,
  spectacle, or ordinary conflict without enough evidence tying the response
  to a social-norm event;
- VLMs confuse an affected participant, victim, camera operator, authority, or
  prank target with an independent bystander;
- source/query lineage can concentrate scripted series, compilations, news,
  police material, or other distributional contamination.

We have responded by building an append-only, audit-gated weak-supervision
program. No retained positive-pillar media is automatically deleted by these
audit rules; the live collector may still purge expendable raw misses according
to its separately configured retention policy. Candidate signals are first run
in shadow, manually audited on frozen source-disjoint cohorts, replicated on
fresh cohorts, and only then allowed to affect retrieval or review order. At
the latest registry checkpoint:

- 28 weak-signal mechanisms are formally registered;
- 10 passed audits for narrowly declared uses such as retrieval, review
  ranking, strict-route exclusion, or pillar rerouting;
- 18 failed transfer and are required to abstain;
- 0 rules are approved for automatic acceptance;
- 0 rules automatically delete or reject media;
- a Snorkel label model has **not** yet been implemented.

The next major step is therefore not to add another opaque “does this video
contain a social norm?” prompt. It is to standardize the already implemented
atomic observations into labeling functions, evaluate their independence and
coverage, combine them in pillar-specific probabilistic label models, and test
whether those combined models outperform the strongest individual signal on
untouched source-disjoint human audits.

---

## 2. The overall scientific goal

### 2.1 The learning problem

The broad objective is to train and evaluate multimodal models that can
recognize and reason about social norms from real or clearly enacted behavior.
The desired model should learn more than a list of abstract norm names. It
should connect:

- an actor or group;
- a temporally bounded behavior;
- an affected person, group, institution, or shared social context;
- a social expectation;
- whether the observed behavior follows, violates, repairs, or illustrates
  that expectation;
- and, when available, how other people respond.

Examples of potentially useful behaviors include interrupting, insulting,
excluding, refusing to help, violating personal space, failing to return a
borrowed item, cutting a queue, creating excessive public noise, littering,
harassing, apologizing, giving credit, helping, intervening, or using an
appropriate conversational response. The project should prioritize ordinary,
recognizable social interactions rather than being dominated by police
confrontations, spectacular accidents, extreme crime, or other highly unusual
contexts.

The long-term task is not merely binary classification. A strong dataset can
support several related tasks:

- whether a norm-relevant event is supported by the evidence;
- whether the event is audiovisually grounded in a specific interval;
- what behavior occurred;
- what expectation applies;
- whether the behavior is a violation, correct example, repair, or contrast;
- what reaction occurred and what it targeted;
- whether the supplied weak label matches the depicted event;
- and whether the clip is clean enough to use without leaking its label.

### 2.2 What “weak supervision” should mean here

The true target—“a social norm violation occurred”—is latent. It is not directly
observable in the same way as a face, a transcript phrase, or a scene cut. A
weak-supervision system should therefore infer the target from observable,
fallible evidence rather than simply rename a VLM’s holistic opinion as a weak
signal.

The project should maintain five distinct concepts:

1. **Latent target:** the proposition we ultimately care about, such as
   `norm_event_supported`.
2. **Atomic observation:** an inspectable claim, such as “a correction phrase
   occurs after the action” or “the saved interval depicts two people in a
   situated exchange.”
3. **Labeling function:** deterministic or model-assisted logic that converts
   one evidence family into `positive`, `negative`, or `abstain`.
4. **Semantic pseudo-labeler:** a fallible LLM/VLM asked to make a higher-level
   judgment from the available evidence. This can be one vote, but not truth.
5. **Human audit label:** a carefully defined expert judgment used for
   calibration and evaluation. It should be stored separately from weak labels.

This distinction matters because directly asking a model
`is_social_norm=yes/no` does not create independent evidence. It asks the model
to estimate the latent target. Such an answer can still be useful as one noisy
annotator, but it must not be counted alongside its own component judgments as
if those were independent votes.

### 2.3 Revised primary targets

The audit program initially used several overly strict or ambiguous composite
targets. The cleaner target decomposition going forward is:

- **`norm_event_supported`**: the available evidence supports a concrete
  social behavior and a socially legible expectation, judgment, or response.
- **`event_audiovisually_grounded`**: the behavior itself is visible or is a
  situated speech act whose participants and context are audiovisually present
  in the saved interval.
- **`label_alignment`**: the assigned normalized behavior, norm, and polarity
  match the event.
- **`reaction_grounded`**: an observable response is temporally and causally
  connected to the behavior.
- **`independent_bystander_signal`**: the responder is a distinct third party or
  organic audience member. This is a valuable witnessed subtype, not the
  universal definition of whether a norm event exists.
- **`clean_training_span`**: the event interval can be separated from the
  textual, spoken, graphic, or reaction signal that supplied the label.

The distinction between `physical_event_visually_depicted` and
`event_audiovisually_grounded` is also necessary. Some social norms concern
situated speech. A threat, insult, refusal, interruption, or apology may not be
recoverable from pixels alone, yet can still be clearly grounded when the
participants, turn-taking, and audio are present in the clip. Conversely,
narration describing an absent event is not audiovisual grounding.

### 2.4 Format policy

The format itself is not a universal accept/reject criterion.

For instructional data, the user’s standing requirement is explicit: **any
format is acceptable as long as there is a real demonstration**. Qualifying
media may include:

- staged live-action role-play;
- organic footage used pedagogically;
- hidden-camera or social-experiment demonstrations;
- film or television excerpts when used as labeled examples;
- animation;
- puppets, toys, or anthropomorphic characters;
- game or simulated scenarios;
- static illustrated stories when the behavior and social setting are actually
  depicted;
- situated dialogue scenes.

Talking heads, slides, explanations, or generic B-roll are not automatically
bad source material, but they are not visual demonstrations unless the target
behavior is also depicted.

For witnessed data, authenticity is more relevant because the weak signal is a
reaction occurring in the scene. Staged material should generally be tagged and
rerouted to instructional review, not discarded.

For commentary, a talking head may provide a perfectly usable text label. It
becomes a visual example only when the underlying event footage is present and
can be isolated.

---

## 3. The three positive pillars and negative controls

### 3.1 Witnessed pillar

#### Intended supervision

Witnessed collection uses a social response as evidence that some preceding or
overlapping conduct was norm-relevant. Useful responses include:

- direct objection: “stop,” “don’t do that,” “leave them alone”;
- correction or sanction;
- protective intervention;
- interposition or separation;
- delegation to a manager, teacher, staff member, or security;
- distraction intended to interrupt a harmful interaction;
- support or assistance to an affected person;
- disapproving evaluation;
- nonverbal gasp, scream, withdrawal, approach, shielding, or coordinated crowd
  reaction.

The minimal evidence graph should be:

1. a localized human action or situated speech event;
2. a response in the same episode;
3. the response occurs after or overlaps the action;
4. the response targets the action or its effects;
5. the response content is normatively informative rather than generic affect;
6. the action interval is recoverable before or separately from the response,
   when leakage-sensitive training requires it.

The responder may be an independent bystander, victim, participant, camera
operator, organic audience, host, or authority. These roles should be recorded
as provenance. Requiring an independent bystander is appropriate for a strict
subcorpus, but it is too narrow as a universal gate for the broader
`norm_event_supported` target.

#### Main failure modes

- generic laughter, surprise, profanity, or excitement without a targeted
  response;
- the detected phrase is spoken by the actor as an excuse rather than by a
  responder;
- the response concerns danger, an accident, an animal, a stunt, gameplay, or
  spectacle rather than a social norm;
- the triggering behavior is off-screen or missing from the saved clip;
- police or authority commands are mistaken for organic social correction;
- a staged series or prank is mistaken for organic witnessed footage;
- a model infers role identity from transcript semantics without binding the
  voice to a visible person;
- the reaction itself leaks into the action training clip.

### 3.2 Instructional pillar

#### Intended supervision

Instructional sources explicitly state or teach a social expectation and
contain one or more demonstrations. A good datum has:

- a specific social behavior grounded in transcript or source context;
- a demonstrated actor;
- a concrete action or situated utterance;
- an affected person, group, or shared social setting;
- a grounded social expectation;
- a bounded demonstration interval;
- alignment between demonstration, normalized behavior, norm, and polarity;
- enough separation from explanation or label-bearing material to create a
  useful training clip.

The saved label may be exact or repairable. A video showing a useful social
behavior should not be discarded merely because the first-pass norm is vague,
such as “respect,” “responsibility,” or “safety.” It can be retained after a
scene-grounded relabel.

#### Main failure modes

- explanation polarity over-extracts narrated statements as demonstrations;
- a business, technical, medical, safety, self-defense, product, or software
  procedure is mislabeled as a social norm;
- the source is on topic but the saved interval shows only a presenter;
- the target event is described rather than enacted;
- animation or B-roll is topically related but does not depict a connected
  actor–action–context episode;
- the assigned label is too abstract or points to the wrong action;
- polarity is wrong—for example, a mocking apology labeled as correct behavior;
- start/end quotes are transcript-grounded but do not bound the visual demo;
- many redundant “demos” are extracted from the same discussion or episode;
- label-bearing narration or captions remain inside the training interval.

### 3.3 Commentary pillar

#### Intended supervision

Commentary provides an explicit normative stance toward conduct. The useful
text-level structure is:

- a concrete occurred event, or a clearly identified repeated event;
- an actor;
- a behavior;
- an affected person or shared social context;
- an identifiable stance source;
- criticism, praise, objection, rule statement, or recommended alternative;
- a transcript-grounded evidence span.

This yields at least a text label. To yield a visual datum, the retained source
must additionally contain a bounded interval in which the event is actually
depicted. The normalized label should describe the event—not the reaction,
consequence, apology, justification, or abstract judgment.

#### Main failure modes

- the source is a podcast, monologue, or news discussion with no event footage;
- the video shows a portrait, slide, unrelated B-roll, or aftermath;
- title or narration refers to an event that never appears;
- the event is present elsewhere in the source but not near the quote;
- the detector extracts the commentator’s reaction rather than the underlying
  action;
- actor and target are reversed;
- normative text, captions, lower-thirds, target circles, or narration leak the
  label into the proposed action clip;
- the source contains staged footage that should be rerouted to instructional
  rather than treated as organic commentary footage.

### 3.4 Negative controls

The corpus includes or plans three useful negative regimes:

1. **No-reaction negatives:** same-source windows away from detected reactions.
   These control for source identity and general visual style.
2. **No-violation negatives:** same-source windows that are reaction-free and
   also pass a reaction-detector cleanliness check.
3. **Null-verified negatives:** mundane-twin queries intended to retrieve
   ordinary non-violation behavior, then explicitly verified.

Historically, tier-1 negative backfill created thousands of clips while
null-verified collection was starved by scheduler priority. The null route also
exposed a batch bug in August 2026, described below. Negative mining should
eventually include visually hard negatives close to positives in embedding
space, while preserving their provenance and avoiding false negatives.

---

## 4. Data model and what counts as a datapoint

### 4.1 Media and labels are separate ledgers

One physical video is not equivalent to one weak label.

- An instructional source may contain several demonstrations.
- A witnessed clip may contain several reaction records.
- A commentary source may contain several normative statements and several
  candidate event windows.

This is why the 2026-08-11 snapshot has 92,286 positive-pillar media files but
156,774 weak-label rows. Corpus reporting must always distinguish:

- source videos;
- derived clips;
- semantic weak-label items;
- event windows;
- manually audited items;
- fully audited media whose every attached label has been reviewed.

### 4.2 Desired final record

A mature training record should contain at least:

- stable item ID and source UID;
- source URL/platform/channel/query provenance;
- pillar and sub-pillar;
- source-cluster and duplicate-cluster IDs;
- exact start/end bounds;
- media hash and transform provenance;
- normalized actor-neutral behavior;
- normalized norm or expectation;
- polarity;
- modality and format tags;
- human/model/weak-signal provenance;
- atomic evidence fields;
- labeling-function votes and abstentions;
- probabilistic posterior(s);
- manual audit state;
- leakage/sanitization state;
- repair/reroute history;
- explicit statement that original media remains preserved.

### 4.3 Leakage-sensitive clipping

Weak supervision is useful only if the input does not trivially expose the
label. The evidence used to certify an event may have to be excluded from the
training interval.

For witnessed clips, the action interval may need to end before the bystander
reaction. For commentary, label-bearing narration, captions, or lower-thirds
may need temporal separation, muting, cropping, or masking. For instructional,
the demonstration should ideally be separable from the teacher’s explicit
explanation. Every transformed artifact must be audited as the exact final MP4;
passing a source-level audit does not certify a crop or recut.

---

## 5. Infrastructure and operational architecture

### 5.1 Hosts

#### sk3

`sk3` is the primary scraper and corpus host. The live repository/data root is:

`/lfs/skampere3/0/alexspan/norm-scraper/`

It runs the main crawl, deferred transcription/detection batches, clip
extraction, audio-event scoring, commentary retention, state database, and most
corpus-wide shadow scoring. The node has eight B200 GPUs, but GPUs 1–4 are
blocked from the scraper batch allocator because they are reserved for other
work. The project must not disturb unrelated research jobs.

#### sk2

`sk2` has been used for managed open-VLM audit batches, including Qwen, GLM,
and Gemma variants. These models are dynamically loaded for a batch and
unloaded afterward. The managed lifecycle prevents idle models from occupying
GPUs indefinitely.

#### sk1

`sk1` has mainly served as an additional controlled host for distributed
YouTube source recovery. Its disk budget and safety margins are more
conservative than sk2/sk3.

### 5.2 Parsimonious server access

The user’s standing operational rule is to minimize SSH activity and avoid
lockouts or rate-triggered access problems. Monitoring and remote actions should
therefore:

- use one connection per check rather than many small SSH calls;
- batch read-only diagnostics;
- prefer append-only on-host monitors over interactive polling;
- run approximately hourly, not continuously;
- avoid login-shell assumptions because AFS home access can fail in
  non-interactive shells;
- invoke Python and ffmpeg by full paths when necessary;
- avoid touching GPU blocklisted devices or unrelated jobs.

The hourly monitor is observational. It records process, inventory, queue,
authentication, disk, and audit-progress state. It is not authorized to promote
labels, delete media, or repeatedly restart jobs.

### 5.3 Collection pipeline

The main pipeline is conceptually:

```text
query scheduler
    -> platform candidate enumeration
    -> global UID deduplication in state.db
    -> metadata/duration/content-policy filtering
    -> source video download
    -> pending_transcribe
    -> WhisperX transcription and alignment
    -> pending_detect
    -> offline Llama-3.3-70B-FP8 detection/routing
    -> witnessed | instructional | commentary | negative | drop/miss
    -> clip/source retention
    -> audio-event scoring for witnessed hits
    -> append-only metadata, label, and audit ledgers
```

The crawl itself is GPU-free. An hourly flocked batch pipeline waits until the
queue is large or stale, selects an allowed GPU with enough free memory, loads
WhisperX, exits and releases memory, then separately loads the 70B detector,
exits and releases memory, and finally runs the small PANNs audio-event model
for new witnessed hits. This design favors high collection throughput without
leaving large inference services resident at 0% utilization.

### 5.4 Dynamic model lifecycle

Open VLM audit servers are no longer intended to be permanent services. The
managed wrapper:

1. accumulates a resumable batch without touching a GPU;
2. waits for a model/profile lock;
3. waits for the configured GPU-memory threshold;
4. starts the exact model and verifies the endpoint;
5. runs the batch;
6. terminates only the process group it started;
7. records cold-start, inference, lease, output, and unload events;
8. releases the GPU even on handled failure or interruption.

This addressed the earlier concern that two GPUs on sk2 were occupied by
models that were often idle.

### 5.5 Platforms and YouTube recovery

The project has used Reddit, Dailymotion, Odysee/LBRY, Rumble, and YouTube with
changing platform reliability. Dailymotion has been a major source but can be
rate-limited. Reddit’s Arctic Shift endpoint has experienced outages or access
constraints. Rumble search was disabled after poor yield and repeated 403s.

YouTube access has changed over time. Three user-provided cookie exports were
installed as separate mode-0600 secrets on sk1, sk2, and sk3 in July 2026.
Each host passed an actual audio-plus-video canary, and the first three bulk
outputs from every host passed a 9/9 manual storage-integrity audit. A frozen
manifest assigned 1,128 missing commentary sources across the three hosts.
These results establish that the credentials worked at launch; they are not a
claim that YouTube access will remain permanently healthy. Authentication
circuit breakers and conservative delays are required.

### 5.6 Commentary source-retention correction

Originally, commentary routing wrote text records and then purged the raw
video. This was a genuine storage defect. At discovery time, roughly 16.8k
discussion records existed, but only 425 unique commentary videos remained
locally recoverable.

The pipeline was corrected so that commentary sources are moved into stable
`data/discussion_video/` storage before expendable raw sidecars are purged. A
resumable backfill was implemented for recoverable URLs, with:

- stable pseudo-random ordering;
- per-host locks;
- ffprobe validation;
- duration/stream validation;
- SHA-256 provenance;
- atomic staged moves;
- append-only attempt ledgers;
- authentication circuit breakers;
- byte and free-disk safety limits.

The latest frozen corpus inventory reports 17,783 retained commentary source
videos, so the commentary pillar is no longer text-only at the storage layer.
However, retaining a source does not certify that its discussed event is
visually present.

---

## 6. Query trajectory and retrieval work completed

### 6.1 Early query drift

The original query generator accumulated tens of thousands of drifted or
low-value queries. Instructional priority increases also flooded the witnessed
route with staged training material. Related-video snowballing amplified bad
series and scripted neighborhoods.

Mitigations included:

- title-level hard negatives for full films, episodes, gameplay, wrestling,
  trailers, and compilations;
- guards against inserting obviously drifted expansion queries;
- channel and series controls;
- family-aware scheduling;
- zero-new cooldowns;
- explicit query-pass ledgers;
- source-failure cooldowns distinct from genuine query saturation;
- source-disjoint audits of new query families;
- query-scope checks on automatically expanded queries;
- reversible policy exclusion rather than deletion.

### 6.2 Typical social-norm query plan

The August trajectory audit examined 36,734 live query rows. It found 30,389
never-used rows, 871 used rows that had never produced a candidate, and an
overweighting toward instructional work that starved witnessed and commentary
retrieval.

The revised canary plan targets ordinary social settings:

- queues;
- doors, elevators, and shared spaces;
- roommates and chores;
- borrowing and returning items;
- transit etiquette;
- public noise;
- cleanup and littering;
- punctuality;
- workplace credit, gossip, and exclusion;
- guest manners;
- customer–worker conduct.

Police, body-camera, traffic-stop, arrest, sheriff, and law-enforcement themes
are explicitly excluded from future discovery. Historical rows are preserved
but reversibly marked out of scheduling scope.

Eighteen manually reviewed canary queries—six per positive pillar—were approved
for retrieval only. Another 36 remain inactive pending visual audits of their
returned videos.

### 6.3 Query scope gate

The automatic scope gate applies to generated `llm_expand`, `exploit`, and
`explore` proposals, not to curated instructional, commentary, taxonomy, or
negative queries. Its frozen wording audit included 90 manually judged
proposals and reported no false allows or false blocks in that cohort.

This gate controls only which queries are scheduled. It never labels a video.
Every proposal and decision is logged with a policy version.

### 6.4 Dailymotion related-video snowball

The Dailymotion related graph initially appeared attractive because children of
a good source might contain similar events. A parent-side development gate
looked promising, but a source-disjoint child transfer failed completely:

- 20 proposed-allow children: 0 confirmed strict witnessed, 1 uncertain,
  19 reject;
- 20 proposed-block children: 0 confirmed strict witnessed, 1 uncertain,
  19 reject.

The allow band was dominated by animal spectacle, driving-fail series,
produced activist confrontations, reality TV, scripted film, news/protest
footage, and game shows. Future Dailymotion related propagation was disabled.
10,735 queued rows were reversibly policy-excluded; no queries or media were
deleted. Reddit’s separate discovery mechanism remained enabled.

---

## 7. Pipeline corrections and engineering work completed

### 7.1 Witnessed precision cleanup

Early audits found substantial scripted, staged, compilation, narration, and
off-topic contamination. A retroactive cleanup added:

- scripted-media hard negatives;
- sports/gameplay/title filters;
- narration-to-commentary rerouting;
- reaction-window merging;
- minimum pre-roll requirements;
- preference for LLM-grounded reaction spans over generic keyword matches;
- title/duration duplicate detection;
- severity and reaction-strength metadata;
- reversible quarantine rather than deletion.

These changes improved retrieval and routing but did not establish a calibrated
automatic keep rule.

### 7.2 Instructional-to-witnessed contamination guard

High-priority `instr_*` queries produced staged training videos that entered the
witnessed corpus. A routing guard was added so non-commentary instructional
query results do not silently become witnessed data. Hundreds of existing
contaminated directories were staged in a reversible quarantine.

### 7.3 SQLite row-access bug

A live detection change used `.get()` on `sqlite3.Row`, which does not support
that method. This stalled detection for several days. The code was corrected to
use bracket access, the backlog was drained, and the incident became an
explicit implementation constraint: do not use `row.get()` on SQLite rows.

### 7.4 Deferred null-route type bug

In August, `save_null_verified()` correctly returned a list of saved windows,
but the database finalizer expected an integer count. Passing the list caused
SQLite to raise a `ProgrammingError` and left the batch pending. The batch
wrapper then continued and falsely printed `pipeline done`.

The fix:

- preserves the extractor’s list-valued API;
- computes the integer count at both routing boundaries;
- passes only the scalar count to SQLite;
- purges raw video only after successful saving;
- checks every GPU-stage exit code;
- stops the pipeline on a failed stage;
- added regression and structural shell tests.

The verified local suite at that checkpoint had 1,445 passing tests.

### 7.5 Commentary retention bug

As described above, commentary source videos were historically purged after
text extraction. The routing code now retains them, and a distributed backfill
was implemented and audited.

### 7.6 VLM input-modality correction

One witnessed audit prompt incorrectly implied that Qwen3-VL could hear the
audio track embedded in an MP4. That model consumes video frames plus separately
provided text; it is not an audio-capable Omni model. The affected experiment
was hard-failed for promotion, every output was retained for error analysis,
and the replacement rubric requires `offscreen_or_unresolved` when ASR speech
cannot be bound to a visible person.

This is a general lesson: model capability claims are part of the audit
contract. Supplying an MP4 does not prove that a VLM consumed audio.

---

## 8. Weak-supervision signals already implemented

### 8.1 Shared atomic social-scope schema

The shared deterministic contract decomposes social scope into four atomic
classifications:

- `actor_kind`;
- `behavior_kind`;
- `affected_context_kind`;
- `expectation_kind`.

Code derives:

- whether a human social actor is grounded;
- whether a concrete behavior is grounded;
- whether an affected person or shared context is grounded;
- whether the evidence is in social-norm scope.

The important design improvement is that the composite is calculated from
enumerated atoms rather than independently supplied as another free-form model
answer. Missing or unresolved evidence yields `uncertain`, which should map to
abstention or review.

### 8.2 Transcript semantic atoms

The transcript scorer classifies:

- interpersonal/public/private/nonhuman actor scope;
- interpersonal, shared-public, conventional, procedural, health/safety,
  belief/ritual, preference/self-improvement, generic, or missing behavior;
- interpersonal, coordination, role, civic, technical, health, preference, or
  identity expectation;
- whether a concrete behavior is grounded;
- whether an affected person/shared setting is grounded;
- whether the assigned norm is exact, broad, wrong-but-relabelable,
  unsupported, or uncertain;
- a normalized behavior;
- an exact evidence quote whose presence is checked against the aligned
  transcript.

These atoms make no visual claim.

### 8.3 Witnessed reaction-language signals

Deterministic reaction phrase families include:

- direct intervention;
- distraction;
- delegation;
- support;
- evaluation/disapproval;
- warning;
- delayed follow-up;
- self-defense/affected-party response;
- reported or narrated speech;
- generic affect only;
- aggregate active-intervention indication.

The clip-wide intervention scan uses these features to generate high-recall
reaction windows. On a 24-video source-disjoint audit it recovered all six
strict bystander-positive sources but proposed 12 false positives: 33.3%
precision and 100% recall. It is therefore useful for localization, not
acceptance.

### 8.4 Witnessed temporal, speaker, role, audio, and face features

Implemented feature groups include:

- explicit diarization availability;
- speaker change at a reaction;
- reactor speaker versus action speaker;
- prior distinct-speaker count;
- overlap at the reaction;
- reaction localization;
- response latency;
- response duration;
- preceding turns in a short temporal window;
- short-response indicator;
- metadata role availability;
- third-party, affected-target, or camera-person role;
- actor/reactor role difference;
- VLM third-party votes and agreement;
- MFCC, pitch, energy, and spectral-change proxies;
- face tracks spanning a reaction boundary;
- head-motion, gaze, and expression change;
- multi-person reaction synchrony;
- scene-cut abstention.

These mechanisms are intentionally named as proxies when they are not true
identity binding. An acoustic change is not diarization; a new voice is not
automatically a bystander; a face-count delta is not a social reaction.

The combined identity-feature experiments did not transfer well enough for
acceptance. Missing diarization caused exact speaker features to abstain, and
whole-scene motion/audio changes were overwhelmed by editing, camera movement,
music, and actor speech.

### 8.5 Witnessed atomic audiovisual contract

The strict shadow contract currently checks:

- `reaction_grounded`;
- `action_before_or_overlaps_response`;
- `response_targets_action`;
- `responder_role`;
- `response_content`;
- `staging`;
- `trigger_kind`.

This is a useful contract for a strict organic independent-bystander subtype.
It must be generalized for the broader primary target so that valid affected-
party, participant, or authority responses can support a norm event without
being mislabeled as independent bystanders.

### 8.6 Cheap visual scene features

The append-only visual baseline scorer includes:

- mean and maximum frame motion;
- histogram change and hard-cut fraction;
- face count and face-presence fraction;
- multiple-face fraction;
- person count and person-presence fraction;
- multiple-person fraction;
- posed-person count;
- person-box area;
- person-count variance;
- relative size of the largest and second-largest person;
- nearest-pair distance;
- close-pair fraction;
- pair overlap;
- wrist proximity to another person;
- person-center motion;
- stable single-person transitions;
- CLIP prompt probabilities;
- X-CLIP prompt probabilities.

CLIP/X-CLIP prompts contrast social interaction, confrontation, role-play,
talking head, news/interview/lecture, text/graphics, generic B-roll, and
off-screen aftermath. These are ranking features, not norm labels.

The corpus-wide cheap activity score has shown some review-ordering value. In
the latest 72-item human calibration, high-activity instructional clips were
more likely to be usable than middle/low clips, while commentary positives
appeared in both high and middle bands. The cells were too small to support a
hard threshold.

### 8.7 Transcript priors for visual presence

Transcript-only priors include:

- reported-speech cues;
- attribution cues;
- retrospective language;
- news framing;
- explicit demo/role-play language;
- “watch/look/here is an example” transitions;
- direct-exchange language;
- visual deixis such as “you can see” or “footage shows.”

These can localize candidate windows or rank them, but they never certify that
the event is visible.

### 8.8 Instructional atomic text contract

The instructional text-side contract checks:

- social behavior domain;
- specific actor role;
- specific behavior;
- affected person/shared context;
- normative obligation;
- literal behavior rather than analogy;
- procedural-skill-only exclusion;
- personal-optimization-only exclusion;
- proposed norm support;
- proposed polarity support;
- normalized behavior and norm;
- grounded behavior and norm quotes.

The contract can route exact labels, repairable labels, retrieval-only items,
or rejects. It explicitly cannot certify a visual demo.

### 8.9 Instructional atomic audiovisual gate

The current instructional social gate uses two label-blind visual observers,
their literal intersection, and the fallible transcript label. Its enumerated
fields include:

- actor kind;
- behavior kind;
- affected-context kind;
- expectation kind;
- depiction support: complete, partial, context-only, explanation-only, none,
  or uncertain;
- whether the grounded quote names a concrete behavior;
- whether that behavior occurs in the depiction;
- whether the interval is presentation-only;
- whether it contains only technical/nonsocial activity;
- exact/broad/relabelable/unsupported label relation;
- normalized behavior;
- failure reason.

Code derives candidate, relabel, recut-review, and semantic/domain-review bands.
The model is not asked for one opaque keep decision.

### 8.10 Instructional exact-demo disposition contract

A separate deterministic contract accepts any performed medium and checks:

- performed behavior;
- actor grounded in the demo;
- target or social context grounded in the demo;
- connected episode;
- label alignment;
- demo medium;
- demo quality;
- exact-video boundary basis;
- polarity;
- valid ordered bounds.

It produces non-destructive dispositions such as candidate, candidate after
relabel, source needs recut, text-only instructional, out-of-scope review, or
uncertain.

### 8.11 Commentary occurred-event text contract

The commentary text contract records:

- social scope;
- occurred-event grounding;
- social actor grounding;
- semantic specificity of the behavior;
- affected person/shared-context grounding;
- normative stance grounding;
- event scope: bounded, repeated, hypothetical, aggregate, ambiguous, or none;
- stance quality: external, contemporaneous bystander, affected-party
  objection, contested, or missing;
- exact behavior and stance quotes;
- normalized behavior and norm.

It can send grounded text items to visual search, definitions to instructional
retrieval, and contested/ambiguous cases to exploratory review. It explicitly
states that transcript evidence cannot certify visibility.

One remaining conceptual cleanup is that this older contract still accepts an
atomic input named `is_social_norm`. That field should be replaced by the
shared actor/behavior/context/expectation decomposition so it is not mistaken
for an independently observable signal.

### 8.12 Commentary visual contract

The visual recovery contract checks:

- text-label decision;
- behavior occurrence: on-camera action, on-camera speech act, off-screen,
  described over B-roll, uncertain, or absent;
- event identity grounding;
- actor/target/context grounding;
- relation between the label span and behavior interval;
- whether normative label content remains inside the action clip;
- transform type;
- sanitization verification;
- demo quality;
- exact-video boundary basis;
- authenticity;
- valid ordered action bounds.

Organic footage can become a commentary visual candidate; staged or animated
footage is rerouted to instructional review. Source-level retrieval does not
imply exact-clip acceptance.

### 8.13 Audio-event channel

PANNs Cnn14 AudioSet scoring provides nonverbal reaction candidates such as
screaming, shouting, gasping, crying, cheering, crowd commotion, slap/smack,
smash/crash, and related events. The model is temporally useful and lightweight
but AudioSet labels are not social-role judgments. Scores are retained as
ranking evidence and can propose additional windows for later semantic review.

### 8.14 Retrieval and provenance signals

Implemented provenance signals include:

- query family and exact query;
- source platform;
- channel;
- related-query ancestry;
- instructional versus commentary provenance;
- title scene cues;
- explanation versus non-explanation polarity;
- retro scan provenance;
- official WWYD source;
- creator/staging cues;
- exact authority/host cues;
- duplicate/series indicators.

These can be extremely useful for retrieval and review ranking, but query or
title text is never itself a positive video label.

---

## 9. Audited weak-signal registry

The registry currently contains 28 named mechanisms. Ten survived for their
narrow declared use; eighteen failed transfer and must abstain. The count is not
the number of raw feature columns—there are many more feature dimensions. It is
the number of formally governed mechanisms whose evidence and allowed use are
recorded.

### 9.1 Audited for declared use

| Rule | Pillar | Allowed use | Important limitation |
| --- | --- | --- | --- |
| `audited_scene_queries_v1` | all | candidate generation | query text is never a label |
| `witnessed_clipwide_intervention_scan_v1` | witnessed | reaction-window proposals | high recall, low precision |
| `witnessed_authority_exact_span_v3` | witnessed | exclude from strict organic route; reroute | does not reject visible scenes |
| `witnessed_creator_staging_title_v2` | witnessed | exclude from organic route; reroute | staging can be useful instructional data |
| `witnessed_official_wwyd_channel_v1` | witnessed | reroute to instructional review | source identity does not certify a demo |
| `instructional_scene_title_cues_v1` | instructional | candidate generation | finds likely sources, not exact demos |
| `instructional_retro_query_source_priority_v1` | instructional | review ranking | nested/correlated with polarity signal |
| `instructional_non_explanation_review_priority_v1` | instructional | review ranking | high recall but unstable precision |
| `commentary_capture_title_event_v1` | commentary | localization candidate/ranking | title action may be metaphorical or absent |
| `commentary_dual_vlm_retrieval_core_v1` | commentary | high-priority source review | source retrieval, not exact clip acceptance |

### 9.2 Failed-transfer mechanisms

The following mechanisms remain implemented and preserved for research/error
analysis, but cannot create an active route or acceptance decision:

| Rule | Pillar | Why it failed or is disabled |
| --- | --- | --- |
| `witnessed_qwen_video_asr_reaction_retrieval_v3` | witnessed | enriched-cohort performance did not transfer to an unbiased 60-source corpus sample |
| `witnessed_identity_multimodal_ensemble_v4` | witnessed | role/identity errors on source-disjoint transfer |
| `instructional_v20_multimodal_rank_high` | instructional | high and low bands had the same 20% visual-demo rate on fresh expansion |
| `instructional_v23_qwen_gemma_consensus` | instructional | 42.9% precision and 37.5% recall on 100-source replication |
| `instructional_title_or_v23_union_v1` | instructional | failed intensive transfer gate |
| `instructional_qwen_blind_episode_relaxed_v1` | instructional | did not transfer as a reliable connected-demo rule |
| `instructional_qwen_demo_consensus_v2` | instructional | failed minimum-yield transfer requirement |
| `instructional_metadata_script_cues_v1` | instructional | metadata/dialogue/action cues did not replicate across cohorts |
| `instructional_visual_title_conjunction_v1` | instructional | failed preregistered precision gate |
| `instructional_demo_adjudicator_v3` | instructional | failed precision gate |
| `commentary_dual_vlm_exact_clip_v1` | commentary | only 2/32 exact transformed clips passed complete contract |
| `commentary_adaptive_edge_crop_v1` | commentary | almost always abstained; both rendered outputs failed |
| `commentary_fixed_caption_crop_v1` | commentary | generic crop did not safely remove leakage |
| `commentary_motion_bbox_crop_v1` | commentary | motion crop did not reliably isolate the event |
| `commentary_multilingual_ocr_mask_v1` | commentary | unsafe full-frame overmasking |
| `commentary_multilingual_ocr_line_mask_v2` | commentary | guarded partial masking still failed |
| `commentary_fixed_tail_trim_v1` | commentary | fixed temporal trimming did not repair exact artifacts reliably |
| `commentary_statement_quote_as_visual_action_v1` | commentary | 0/30 raw phrases were acceptable unchanged as visual action labels |

Failed mechanisms are scientifically valuable. They identify what does not
generalize and prevent repeated investment in superficially promising but
correlated or distribution-specific signals.

### 9.3 Zero acceptance gates is intentional

The registry sets `acceptance_gate_rules` to zero. Combined records preserve
review routes and strict-route exclusions, while `acceptance_label` and
`corpus_disposition` remain null. `delete_media` remains false.

This does not mean the project has made no progress. It means the evidence has
supported better retrieval, prioritization, rerouting, failure diagnosis, and
safe preservation—but not yet a sufficiently validated automatic keep
decision.

---

## 10. Manual audit program completed so far

### 10.1 Audit methodology

The strongest audits use the following procedure:

1. Define a target and exact rule before looking at the evaluation outputs.
2. Freeze a source-disjoint sample.
3. Exclude prior manually reviewed source UIDs.
4. Review temporal frames or storyboards without revealing the weak label when
   possible.
5. Escalate uncertain items to dense multi-frame or exact-video review.
6. Reveal transcript, label, query, and metadata only after the visual judgment
   is frozen.
7. Judge semantic alignment, polarity, provenance, reroute, and repair needs.
8. Audit the exact transformed MP4 when a crop, trim, mute, or recut is proposed.
9. Report precision, recall, coverage, uncertainty, Wilson intervals, and
   failure strata.
10. Replicate on a fresh source-disjoint cohort before promotion.

This is deliberately stricter than spot-checking a midpoint frame. A single
frame can miss brief demonstrations or misrepresent edited videos.

### 10.2 Corpus-wide audit coverage

The 2026-08-11 frozen inventory reports:

- 91,251 of 92,286 positive media files have cheap visual coverage: **98.9%**;
- 696 weak-label rows have a manual strict judgment;
- every attached label is manually complete for 488 positive media files;
- 140,972 media-backed weak-label rows remain in the strict audit queue.

The remaining queue by label type is large because commentary and witnessed
sources can contribute multiple labels. Manual audit coverage is therefore
valuable but still far below complete corpus adjudication.

### 10.3 Why historical manual rates are not corpus precision estimates

Many audits deliberately targeted difficult examples, proposed-rule outputs,
suspected contamination, or repair candidates. They are not simple random
samples from the entire library. Reported accept/reject counts should be used
to diagnose mechanisms and estimate performance only within the frozen cohort
design—not as universal corpus survival rates.

### 10.4 Latest 72-item human calibration

Two source-disjoint waves sampled 72 items across pillar, old/new cheap-feature
coverage, and low/middle/high activity bands. Every item received blind temporal
review; uncertain cases received dense follow-up; all items received post-reveal
semantic and routing review.

| Pillar | Strict or recoverable | Reject | Interpretation |
| --- | ---: | ---: | --- |
| Instructional | 5/24 | 19/24 | 2 clean; 3 need trim or relabel; multiple formats can pass |
| Witnessed under strict independent-bystander contract | 0/24 | 24/24 | 20/24 still contain a salvageable social scene; the strict subtype was too narrow as a universal target |
| Commentary visual | 7/24 | 17/24 | many rejects remain useful as text; 3/7 visual positives need repair |

The witnessed result must be interpreted carefully. It does **not** mean that
no witnessed videos contain social norm events. It means none passed the much
narrower organic independent-bystander contract in that cohort. Twenty of 24
still contained a potentially useful scene. This motivated the target reframe
from “perfectly identify an independent bystander” to “support that a concrete
norm event occurred,” while retaining role and reaction subtype as auxiliary
labels.

### 10.5 Qwen calibration on 36 items

Qwen3-VL-8B was dynamically loaded for one batch, produced 36 outputs, and was
unloaded. Every response was manually reviewed:

- 16 accurate;
- 9 partially accurate;
- 11 materially inaccurate.

On that cohort:

- instructional strict-scene ranking: precision 1.00, recall 0.33;
- commentary visual ranking: precision 1.00, recall 0.40;
- strict witnessed selection: five false positives and zero true positives.

The witnessed errors were systematic role errors. The model confused affected
targets, camera participants, prank targets, and animal-directed reactions
with independent bystanders. Qwen can rank some instructional and commentary
candidates but is not approved as a witnessed keep rule.

### 10.6 Instructional audit findings

Important instructional results include:

- Initial transcript quote grounding was structurally high, but visual content
  was much less reliable.
- Explanation polarity was a major leak source; many explanation records had no
  demonstration.
- A fresh 36-source V22 audit found ten complete label-aligned demonstrations.
- Qwen/Gemma consensus initially looked promising at 4 TP and 1 FP, but a
  preregistered 100-source replication fell to 6 TP, 8 FP, and 10 FN.
- Scene-oriented title cues retrieved visual demonstrations at 83.3% precision
  but only 31.3% recall across 185 audited sources. Exact label alignment was
  lower, so titles remain source-level retrieval cues.
- Non-explanation polarity is useful for review ordering but not automatic
  keeping. Across two source-disjoint cohorts, it captured 56/60 visual demos
  but its precision varied substantially.
- Retro-scan provenance and non-explanation polarity are correlated; they must
  not be treated as independent Snorkel votes.
- Cheap multimodal activity/ranking rules repeatedly failed to distinguish
  actual demonstrations from illustrative B-roll, tutorials, montages, and
  unrelated animation.

### 10.7 Witnessed audit findings

Important witnessed results include:

- Reaction phrase localization can achieve high recall but low precision.
- Whole-frame and whole-audio before/after changes do not approximate a
  controlled bystander-reaction experiment on edited web video.
- Exact diarization correctly abstains when speaker IDs are absent.
- Acoustic speaker-change proxies fire on ordinary dialogue and cannot identify
  a bystander.
- Silent VLMs cannot bind ASR speech to a visible speaker.
- VLMs frequently confuse victims, participants, hosts, authorities, and camera
  operators with independent bystanders.
- Creator/staging and authority cues are useful for rerouting narrow organic
  witnessed candidates, not for discarding social scenes.
- The Dailymotion related-video graph failed child-level transfer and was
  disabled.
- Many clips rejected by the strict witnessed subtype may still be useful as
  instructional demonstrations or general norm-event examples.

### 10.8 Commentary audit findings

Important commentary results include:

- Commentary text often contains a valid normative stance even when no visual
  event is present.
- Raw extracted statement phrases are poor visual action labels. In a fresh
  30-row audit, none was acceptable unchanged; 16 sources still warranted
  visual search after actor–action–target relabeling.
- Title/capture cues can retrieve sources containing event footage.
- Dual-VLM agreement transferred well for **source ranking**: seven selected
  sources in a source-disjoint cohort were all manually usable, but the method
  missed 11 of 18 usable sources.
- Source-level precision did not transfer to exact clips. Only 2/32 bounded
  transformed clips passed the full action, temporal, audio, leakage, and
  semantic contract.
- News graphics and label-bearing text appeared in 23/32 exact proposals.
- Fixed trimming, cropping, motion boxes, and OCR masking did not provide a
  general automatic repair mechanism.
- Exact manual repair can work on individual sources, but every resulting
  artifact must be audited.

---

## 11. What has helped and what has not

### 11.1 Mechanisms that have clearly helped

- retaining all source media and using append-only ledgers;
- separating media inventory from label inventory;
- source-disjoint audit sampling;
- blind visual review before semantic reveal;
- exact-artifact audit after transforms;
- query-family balancing and zero-new cooldowns;
- explicit police/law-enforcement query exclusion;
- source/channel/staging/authority rerouting;
- commentary source-video retention and backfill;
- dynamic model load/unload;
- reaction-window proposal generation;
- title/query cues for candidate retrieval;
- non-explanation/retro provenance for instructional review order;
- dual-VLM agreement for commentary source ranking;
- atomic actor/behavior/context/expectation schemas;
- preserving repairable labels instead of forcing clean/reject only;
- named failure modes and abstention;
- failed-transfer registry enforcement.

### 11.2 Mechanisms that have not yet helped enough

- a direct holistic “is this a social norm?” model answer;
- whole-scene motion or audio change as a bystander detector;
- acoustic speaker change as role identity;
- face count or multiple-person detection as proof of social interaction;
- pose alone as proof of a norm event;
- generic CLIP/X-CLIP scene scores as acceptance rules;
- silent VLM plus ASR for visible speaker identity;
- open-VLM consensus without fresh replication;
- title/query evidence as a label;
- explanation polarity as a demonstration;
- fixed crops, tail trims, or OCR masks as universal commentary repair;
- related-video snowball propagation without child-level audit;
- strict independent-bystander identity as the universal witnessed target.

### 11.3 The bottom-line interpretation

The current features are not useless. They are best at:

- retrieving candidates;
- proposing temporal windows;
- prioritizing manual/VLM review;
- identifying obvious source families;
- detecting likely reroutes;
- identifying disagreement and hard cases;
- preserving provenance;
- avoiding repeated known contamination.

They have not yet produced a proven, high-precision, high-recall automatic keep
decision. That is why a probabilistic combiner and a more carefully defined
target are the next logical steps.

---

## 12. Snorkel-style combination: what remains to implement

### 12.1 Current state

There is no actual Snorkel `LabelModel`, LF matrix, or trained probabilistic
combiner in the repository. The code contains many atomic scorers, contracts,
and audited mechanisms, plus one append-only bridge that combines registered
shadow-score records. That bridge enforces allowed uses; it does not learn LF
accuracies or output a posterior probability of the latent target.

### 12.2 Why Snorkel is appropriate

Each candidate has incomplete and conflicting evidence. A labeling function
can emit:

- `+1`: evidence supporting the target;
- `-1`: evidence against the target;
- `0`: abstain because the evidence family is unavailable or unresolved.

A label model estimates the reliability and dependence structure of the
labeling functions and produces a probabilistic label. This is preferable to
hard-coded vote counting because coverage and error rates differ greatly.

### 12.3 Required LF interface

Every LF should produce a standardized record such as:

```json
{
  "item_id": "instructional:youtube__...:2",
  "pillar": "instructional",
  "target": "event_audiovisually_grounded",
  "lf_id": "instructional_non_explanation_v1",
  "family": "transcript_structure",
  "vote": 1,
  "confidence": 0.72,
  "abstain_reason": null,
  "evidence": {"polarity": "violation"},
  "model_or_rule_version": "...",
  "source_artifact_sha256": "..."
}
```

The key requirements are:

- stable LF IDs and versions;
- one explicit target per vote;
- evidence-family grouping;
- explicit abstention;
- immutable provenance;
- no hidden mutation of corpus labels;
- no direct use of query/title as truth;
- clear distinction between deterministic, model-assisted, and human labels.

### 12.4 Separate label models by pillar and target

One universal model would mix incompatible evidence processes. At minimum,
build separate models for:

#### Witnessed

- `norm_event_supported`;
- `reaction_grounded`;
- `independent_bystander_signal` as an optional subtype;
- `event_audiovisually_grounded`;
- `clean_pre_reaction_span`.

#### Instructional

- `explicit_social_rule_grounded`;
- `demonstration_present`;
- `event_audiovisually_grounded`;
- `label_alignment`;
- `clean_demo_span`.

#### Commentary

- `commentary_text_label_supported`;
- `occurred_event_supported`;
- `event_present_in_source`;
- `event_audiovisually_grounded`;
- `clean_action_span`.

### 12.5 Evidence-family grouping

The label model must not count correlated paraphrases as independent evidence.
Suggested families are:

- event/action;
- reaction;
- temporal/causal grounding;
- actor/role binding;
- social context;
- explicit semantics;
- visual depiction;
- source/format provenance;
- contradiction/hard-negative;
- semantic VLM/LLM judgment;
- retrieval lineage.

For example, “contains stop,” “contains an objection phrase,” an LLM’s
“targeted objection” answer, and a VLM+ASR “negative reaction” answer may all be
driven by the same utterance. They should be consolidated, modeled as dependent,
or constrained so they cannot create four independent votes.

### 12.6 Gates versus probabilistic votes

Some conditions should remain deterministic eligibility gates rather than
being outvoted:

- media exists and decodes;
- bounds are ordered and valid;
- an instructional visual datum has a demonstration interval;
- a commentary visual datum contains the event, not only commentary;
- a transformed artifact has been verified when sanitization is required;
- missing evidence yields review/abstention rather than invented certainty.

Twenty transcript votes should not compensate for the complete absence of a
visual demonstration.

### 12.7 Training and evaluation procedure

For each pillar/target:

1. Freeze a sufficiently large source-disjoint human gold set.
2. Split by source video, channel, duplicate cluster, and query ancestry.
3. Materialize every candidate LF without fitting thresholds on test labels.
4. Report LF coverage, overlap, conflict, precision, recall, and correlation.
5. Establish simple baselines: majority vote, strongest single LF, logistic
   regression on audited training labels, and calibrated semantic-model score.
6. Fit the unsupervised or semi-supervised label model only on the designated
   training population.
7. Calibrate posterior probabilities on a held-out calibration set.
8. Evaluate on a completely untouched source-disjoint test set.
9. Report precision–recall curves, calibration error, subgroup performance,
   and Wilson intervals at proposed operating points.
10. Manually audit every predicted positive plus a random sample of predicted
    negatives before any promotion.

The label model is successful only if it beats the strongest individual LF and
simple baselines. Agreement alone is not evidence of truth.

### 12.8 Output policy

Initial Snorkel outputs should be append-only shadow probabilities:

- high-confidence candidate;
- disagreement/manual-review priority;
- likely named failure/reroute;
- insufficient evidence/abstain.

They should not automatically delete, reject, or overwrite the corpus. A later
promotion decision should require fresh prospective audits at the exact chosen
thresholds.

---

## 13. Full-corpus backfill status and remaining scale problem

### 13.1 Completed coverage

The cheap visual backfill is nearly complete: 98.9% of positive media had cheap
visual coverage in the latest frozen snapshot. Instructional and witnessed
also have substantial pillar-specific shadow-score coverage.

### 13.2 Remaining strict queue

The strict queue still contains 140,972 media-backed weak-label rows. This is
far too large for one-by-one closed-model review at high cost. A scalable plan
must combine:

- cheap feature ranking;
- open-VLM batch scoring;
- selective closed-model/manual adjudication;
- uncertainty and disagreement sampling;
- source/cluster-aware audit design;
- resumable, append-only shards;
- dynamic GPU leases;
- coverage accounting by both label and physical media.

### 13.3 Do not discard unreviewed media

The unreviewed remainder is unlabeled with respect to the strict target, not
negative. It should remain in the corpus. Positive-unlabeled methods and review
bands are preferable to treating every unapproved item as a reject.

### 13.4 Recommended audit sampling mixture

Each recurring audit wave should combine:

- a uniform random source sample for population estimation;
- a high-score sample to estimate candidate precision;
- a low-score sample to estimate false-negative risk;
- a disagreement sample across evidence families;
- a fresh-query sample to detect retrieval drift;
- a repair-candidate sample;
- source/channel caps to prevent series domination;
- unseen duplicate/query-ancestry clusters.

Manual judgments should be synchronized into compact ledgers on sk3, while
local rendered frames and large disposable artifacts can be deleted after
hashes and decisions are preserved.

---

## 14. Open-source and closed-model roles

### 14.1 What open VLMs can do now

Open VLMs have shown useful but limited ability to:

- rank obvious instructional demonstrations;
- identify some commentary sources containing event footage;
- describe visible actors and actions;
- distinguish some presentation-only or graphic-heavy clips;
- propose temporal candidates;
- provide a second independent observation for audit triage.

They have not reliably solved:

- brief demonstrations;
- actor–speaker identity binding;
- independent bystander versus affected participant;
- exact norm/polarity alignment;
- precise temporal boundaries;
- absence of label leakage;
- source-to-clip transfer;
- robust performance across unseen retrieval distributions.

### 14.2 Closed-model/manual audit role

Interactive GPT-5.6-sol-class review is available locally through the user’s
authenticated environment and can provide high-quality manual semantic audits.
Unattended scheduled execution on sk3 has historically lacked the necessary
remote credential/endpoint. Therefore “manual Sol audit” and “automated remote
closed-model approval” must not be conflated.

Closed-model judgments should be used to:

- create and adjudicate gold audits;
- analyze open-model errors;
- define better atomic rubrics;
- verify exact candidate artifacts;
- sample and monitor the label model;
- distill high-quality labels into an open critic.

They should not silently become unbounded production labels without explicit
provenance and calibration.

### 14.3 Desired open-model research path

1. Benchmark each open VLM on the same frozen source-disjoint items.
2. Audit every output, not just final decisions.
3. Record modality actually consumed.
4. Ask atomic questions rather than a holistic norm question.
5. Test source-level retrieval separately from exact-clip certification.
6. Evaluate disagreement with text and cheap visual signals.
7. Distill only after repeated cross-source transfer.
8. Dynamically load and unload models in large batches.

---

## 15. Post-training strategies left to implement

Weak supervision need not end with a label model. Several post-training
mechanisms could improve data quality.

### 15.1 Multi-head quality critic

Train a smaller audiovisual/text model on the accumulated manual atomic labels.
Its heads should predict inspectable properties rather than one vague final
label:

- behavior present;
- event depicted;
- situated speech present;
- social context grounded;
- reaction present;
- response targets action;
- role subtype;
- demo–label alignment;
- polarity alignment;
- presentation-only;
- label leakage;
- repair needed.

The critic can rank the corpus and supply additional LFs. It should not be
trained on unaudited pseudo-labels alone.

### 15.2 Pairwise ranking

Construct within-source pairs in which an audited event interval should rank
above:

- a nearby talking-head segment;
- an off-screen description;
- an aftermath;
- an unrelated scene;
- a wrong-role reaction;
- a technical procedure;
- a label-leaking interval.

Within-source pairs reduce reliance on channel and genre shortcuts.

### 15.3 Positive-unlabeled learning

Treat strict manual positives as positive and the rest as unlabeled. Use only
explicit, high-confidence hard negatives as negatives. This is more faithful
than treating every nonaccepted corpus item as false.

### 15.4 Cross-modal disagreement mining

High-value audit cases include:

- transcript asserts an occurred event but vision sees presentation only;
- vision sees a social interaction but the label is unrelated;
- reaction phrase exists but no targetable preceding action is visible;
- title claims footage but video shows commentary;
- two VLMs disagree;
- label model has high entropy;
- cheap features and semantic model strongly disagree.

These cases are likely to improve the next generation of rules more than
randomly auditing only obvious positives.

### 15.5 Training-dynamics triage

After training a preliminary downstream model, audit examples with:

- persistent high loss;
- prediction instability;
- high influence;
- disagreement across epochs;
- anomalously easy performance suggesting leakage;
- strong train/test cluster similarity.

These signals should schedule review. They should not automatically delete
data.

### 15.6 Cluster-aware deduplication and split design

Near duplicates and series can make evaluation look artificially good. Build
clusters from:

- perceptual video hashes;
- audio fingerprints;
- transcript embeddings;
- normalized titles;
- channel/source identity;
- related-query ancestry;
- repeated visual intros/outros;
- episode and compilation structure.

Train, validation, and test splits must be grouped by these clusters.

---

## 16. Remaining implementation roadmap

### Priority 0: preserve collection health

- Keep the GPU-free crawl and deferred batch pipeline healthy.
- Ensure commentary source retention remains enabled.
- Continue hourly, low-frequency observational monitoring.
- Maintain YouTube authentication circuit breakers and per-host ledgers.
- Track collection by platform, pillar, source, clip, and label—not only total
  directories.
- Detect stalled `pending_transcribe` or `pending_detect` queues without
  repeatedly hitting SSH.
- Continue dynamic load/unload for large models.
- Never touch blocklisted GPUs or unrelated jobs.

### Priority 1: finalize the target ontology

- Replace ambiguous composites with the revised targets in Section 2.3.
- Generalize witnessed acceptance away from mandatory independent-bystander
  identity.
- Retain `independent_bystander_signal` as a high-value subtype.
- Distinguish physical visual depiction, situated audiovisual speech, and
  description-only.
- Refactor commentary’s direct `is_social_norm` input into the shared atomic
  scope schema.
- Create one versioned schema used by human audits, LFs, VLMs, and downstream
  training.

### Priority 2: standardize labeling functions

- Implement the common LF record interface.
- Wrap every audited active signal as `+1/-1/abstain` for one target.
- Group correlated signals into families.
- Make failed-transfer signals abstain by construction.
- Store continuous scores separately from thresholded votes.
- Add unit tests for vote, abstention, invalid inputs, missing evidence,
  provenance, and versioning.
- Generate LF coverage/overlap/conflict reports.

### Priority 3: build the Snorkel experiments

- Add the Snorkel dependency or implement an equivalent auditable generative
  label model.
- Build separate label matrices per pillar and target.
- Compare independent-family and dependency-aware models.
- Use human audits for calibration and model selection.
- Benchmark against majority vote, strongest LF, and supervised baselines.
- Produce posterior probabilities and calibrated bands only in shadow.
- Audit all high-confidence positives and random negatives before promotion.

### Priority 4: expand human gold strategically

- Build larger random, source-disjoint cohorts for population estimates.
- Ensure enough true witnessed reaction positives to estimate precision and
  recall.
- Include both independent-bystander and affected-party/participant responses.
- Balance instructional formats and norm categories.
- Separate commentary text quality from visual-event quality.
- Double-adjudicate a subset and report human agreement.
- Preserve exact rationale and atomic evidence, not only final labels.

### Priority 5: complete scalable corpus backfill

- Finish any remaining cheap-score gaps.
- Run open VLMs in resumable, dynamically leased batches.
- Prioritize high-score and disagreement bands.
- Maintain random controls to estimate missed positives.
- Write all scores to compact server-side ledgers.
- Avoid storing large redundant local frames after judgments are synchronized.
- Report coverage by label and by physical media.

### Priority 6: improve event localization and repairs

#### Witnessed

- bind actions, voices, and visible people when possible;
- use audio-capable models when claiming audiovisual speaker evidence;
- find the best reaction, not only the originally detected phrase;
- recover clean action-only pre-response intervals;
- retain participant/victim reactions as general norm evidence;
- reroute clearly staged scenes to instructional.

#### Instructional

- locate the actual demonstrated episode, not merely quote bounds;
- identify brief situated-speech demos;
- cluster redundant demos per source;
- normalize vague labels into concrete behavior descriptions;
- cut or sanitize explanation leakage;
- audit exact recuts.

#### Commentary

- search wider than the quote window;
- normalize actor–action–target before visual search;
- use temporal hierarchy: source overview, candidate scene, dense exact window;
- separate source retrieval from clip acceptance;
- audit captions, lower-thirds, narration, and target graphics;
- reroute staged/animated scenes to instructional.

### Priority 7: improve retrieval based on audited yield

- Run and audit the 18 active ordinary-norm canaries.
- Keep the remaining 36 proposals inactive until their outputs are reviewed.
- Generate new queries only inside the audited ordinary-social scope.
- Preserve police/law-enforcement exclusion.
- Track query yield through exact usable clips, not merely downloads or
  detector routes.
- Use source/channel caps and diversity objectives.
- Do not re-enable Dailymotion snowball without a new child-level transfer
  audit.

### Priority 8: negatives and evaluation corpus

- Continue no-reaction and no-violation negative generation.
- Revive null-verified collection with fair scheduling, now that the null route
  is fixed.
- Audit negatives for hidden violations or reactions.
- Mine hard negatives close to positives in embedding space.
- Construct evaluation splits with strict cluster separation.
- Measure both event presence and weak-signal presence to understand selection
  bias.

### Priority 9: downstream critic and dataset release

- Train the multi-head quality critic.
- Perform pairwise ranking and PU-learning experiments.
- Use training dynamics for audit scheduling.
- Freeze versioned corpus manifests.
- Document licensing/source provenance and retention policy.
- Export separate dataset views:
  - text commentary labels;
  - visual commentary events;
  - instructional demonstrations;
  - broad witnessed norm events;
  - strict independent-bystander subset;
  - clean action-only leakage-sensitive clips;
  - negatives and uncertain/review queues.

---

## 17. Proposed milestones and promotion criteria

### Milestone A: unified atomic schema

**Deliverables**

- one versioned schema for shared and pillar-specific atoms;
- converters for existing audit ledgers;
- removal of opaque composite questions where atomic derivation is possible;
- complete unit tests.

**Pass condition**

- two reviewers can apply the schema to a frozen mixed-pillar cohort with
  acceptable agreement;
- every final target is deterministically derived or explicitly marked as a
  semantic pseudo-label.

### Milestone B: LF matrix and diagnostics

**Deliverables**

- standardized LF outputs;
- pillar-specific matrices;
- coverage, overlap, conflict, correlation, and per-LF audit reports;
- active/failed registry integration.

**Pass condition**

- no failed-transfer rule can emit a live vote;
- no correlated duplicate family is accidentally counted multiple times;
- all votes have reproducible provenance.

### Milestone C: Snorkel shadow model

**Deliverables**

- separate label models by pillar/target;
- posterior scores;
- baseline comparison;
- calibrated operating bands;
- untouched source-disjoint test results.

**Pass condition**

- the combined model improves meaningfully over the strongest individual LF;
- calibration is acceptable;
- subgroup failures are understood;
- high-confidence positives pass complete manual review at the preregistered
  threshold;
- sampled negatives show acceptable missed-positive risk.

### Milestone D: corpus-scale shadow backfill

**Deliverables**

- probabilities for all media-backed weak labels with sufficient evidence;
- explicit abstentions for unsupported rows;
- coverage dashboards;
- compact server-side ledgers;
- no corpus mutation.

**Pass condition**

- all shards are resumable and hash-stable;
- counts reconcile with the media and label manifests;
- random audits match expected calibration within uncertainty.

### Milestone E: limited production promotion

**Deliverables**

- one or more approved uses such as review priority, reroute, or high-confidence
  candidate view;
- rollback instructions;
- prospective monitoring.

**Pass condition**

- fresh post-deployment audits reproduce the preregistered precision/recall;
- no destructive action occurs;
- low-confidence media remains preserved;
- distribution drift triggers abstention or rollback.

---

## 18. Major unresolved research questions

1. What is the right breadth of the witnessed target: any socially informative
   response, or a separate strict independent-bystander subset in addition to
   the broad target?
2. How much situated speech can be reliably grounded with video frames, ASR,
   and speaker information without an audio-capable multimodal model?
3. Can an open VLM reliably detect demonstrations across live action,
   animation, puppets, games, and static illustrated stories without learning
   format shortcuts?
4. How often can valid commentary text labels be converted into visual events,
   and what localization hierarchy gives acceptable yield?
5. How should label leakage be defined for each downstream task? A caption may
   be unacceptable for evaluation but acceptable for weakly supervised
   pretraining.
6. How much of the corpus is repairable through relabeling or recutting rather
   than rejection?
7. Can role binding be improved using true audiovisual active-speaker models,
   person tracking, or audio-capable VLMs?
8. Which evidence families provide genuinely independent information in a
   Snorkel model?
9. How biased is the witnessed corpus toward severe, public, unambiguous, and
   low-cost-to-confront violations?
10. Which ordinary social norms remain underrepresented because web search and
    reaction-based supervision favor spectacle?
11. What audit budget is required for stable precision estimates across norm,
    source, language, format, and platform strata?
12. Can a multi-head critic trained on atomic labels generalize better than
    direct VLM prompting?

---

## 19. Risks and safeguards

### 19.1 Statistical risks

- targeted audits can be mistaken for population estimates;
- duplicate sources inflate apparent sample size;
- source/channel leakage inflates held-out performance;
- correlated LFs produce false confidence;
- low positive prevalence makes precision estimates unstable;
- review-ranking audits do not establish recall;
- initial VLM success may disappear on fresh retrieval distributions.

### 19.2 Semantic risks

- abstract norms are assigned without concrete actions;
- legality, safety, technical correctness, or personal preference is mistaken
  for a social norm;
- the response is labeled instead of the triggering event;
- roles are inferred rather than grounded;
- a valid scene receives the wrong polarity;
- source-level relevance is mistaken for exact-clip alignment.

### 19.3 Visual and temporal risks

- midpoint frames miss brief actions;
- edited videos reorder action and reaction;
- the event appears elsewhere in a source;
- captions or narration leak the label;
- transforms remove essential context or retain hidden leakage;
- low-resolution or corrupted video produces false absence.

### 19.4 Operational risks

- YouTube cookies expire or trigger bot protection;
- excessive SSH or platform requests cause lockouts;
- shared GPUs are occupied or accidentally touched;
- long-running model servers remain idle-resident;
- a batch wrapper hides a failed stage;
- AFS login paths fail in noninteractive sessions;
- local audit artifacts consume excessive disk;
- node-local backfill outputs are mistaken for consolidated corpus files.

### 19.5 Existing safeguards

- append-only ledgers;
- no automatic deletion;
- reversible policy exclusions and quarantine;
- source-disjoint sampling;
- exact artifact hashes;
- failed-transfer abstention;
- managed model lifecycle;
- GPU blocklists;
- queue thresholds and disk guards;
- authentication circuit breakers;
- hourly low-frequency monitoring;
- explicit test coverage;
- preservation of original source media where available.

---

## 20. Repository map for future work

### Production pipeline

- `src/search_loop.py`: crawl loop and shared routing.
- `src/batch_transcribe.py`: deferred transcription.
- `src/batch_detect.py`: offline detector and routing.
- `src/llm_detect.py`: witnessed/commentary and instructional detector prompts.
- `src/clip_extract.py`: clip saving, negatives, commentary retention,
  instructional saving.
- `src/detect_reactions.py`: reaction lexicon detector.
- `src/audio_events.py` and batch modules: nonverbal audio-event channel.
- `src/state.py`: query and video state database.
- `src/query_scope.py`: automatic query scope policy.
- `batch_pipeline.sh`: hourly GPU batch lifecycle.

### Atomic contracts and scorers

- `scripts/weak_label_contract.py`: shared atomic social scope.
- `scripts/witnessed_reaction_av_contract.py`: strict witnessed reaction graph.
- `scripts/witnessed_reaction_text_features.py`: intervention language families.
- `scripts/witnessed_reaction_identity_features.py`: speaker, role, acoustic,
  face, and synchrony proxies.
- `scripts/instructional_social_behavior_contract_v5.py`: text-side social
  behavior contract.
- `scripts/score_instructional_social_gate.py`: atomic instructional AV gate.
- `scripts/instructional_demo_contract.py`: exact instructional disposition.
- `scripts/commentary_occurred_event_contract_v3.py`: commentary text event
  contract.
- `scripts/commentary_visual_contract.py`: commentary visual recovery contract.
- `scripts/score_visual_scene_baselines.py`: motion, face, pose, CLIP, and X-CLIP
  features.
- `scripts/score_transcript_scene_priors.py`: transcript-only depiction priors.
- `scripts/score_transcript_visual_deixis.py`: candidate localization from visual
  reference language.

### Governance and audit

- `config/audited_weak_signals_v1.json`: formal signal registry.
- `scripts/weak_signal_registry.py`: registry validation and enforcement.
- `scripts/combine_audited_shadow_scores.py`: append-only governed score bridge.
- `docs/audited_weak_signal_registry_v1.md`: human-readable registry summary.
- `docs/weak_supervision_visual_audit.md`: detailed audit chronology.
- `docs/witnessed_reaction_signal_cycles.md`: witnessed feature experiments.
- `docs/weak_supervision_post_training_v1.md`: planned label-model and
  post-training architecture.
- `docs/managed_audit_vllm.md`: dynamic audit-model lifecycle.
- `audit_runs/`: frozen manifests, model outputs, ledgers, exact artifacts,
  evaluations, and reports.

### Current authoritative frozen snapshots

- `audit_runs/20260811_strict_audit_backfill_v1/final_summary_20260811.json`
- `audit_runs/20260811_strict_audit_calibration_v2/AUDIT_REPORT.md`
- `config/audited_weak_signals_v1.json`

These should be treated as dated snapshots. Any live corpus claim after
2026-08-11 should be based on a new synchronized inventory rather than silently
assuming the counts are unchanged.

---

## Appendix A. Data and artifact location catalog

This appendix is the location index for the project. It distinguishes the live
bulk corpus, the local development/audit checkout, node-local recovery data,
derived outputs, operational state, model/runtime assets, and ephemeral scratch.

The paths below are based on checked-in configuration, scripts, and frozen audit
reports. They were not refreshed with an interactive server query on
2026-08-17. A path described as “expected” or “configured” should therefore be
verified through one parsimonious inventory command before a destructive or
high-cost operation.

### A.1 Root directories

| Purpose | Host | Location | Authority/status |
| --- | --- | --- | --- |
| Local development and compact audit checkout | local macOS | `/Users/spangher/Projects/stanford-research/social-norms/` | Current workspace for code, docs, tests, compact audit ledgers, and selected audit artifacts; not the bulk corpus |
| Live scraper and primary corpus | sk3 | `/lfs/skampere3/0/alexspan/norm-scraper/` | Canonical live collection root |
| Convenience symlink to live scraper | sk3 | `~/norm-scraper` | Historically points to the canonical sk3 root; AFS home access may fail in noninteractive shells |
| Old quiet crawler | sk3 | repository commonly referred to as `norm-scraper-quiet` | Old code; explicitly out of scope and must not be modified or pruned |
| Managed audit workspace | sk2 | `/lfs/skampere2/0/alexspan/visual_audit/` | Managed open-VLM batch workspace and lifecycle logs |
| Qwen experiment launchers | sk2 | `/lfs/skampere2/0/alexspan/prefix5_lane.sh` and `/lfs/skampere2/0/alexspan/budgetmatch_lane.sh` | Node-local launchers mirrored under local `ops/sk2/` |
| Distributed YouTube worker storage | sk1/sk2/sk3 | `<worker --storage-root>/data/discussion_video/` | The worker script derives this path from an explicit launch argument; exact sk1/sk2 absolute roots are not recorded in the checked-in launch audit |

### A.2 Canonical sk3 data tree

Unless otherwise noted, prepend the live root
`/lfs/skampere3/0/alexspan/norm-scraper/` to each relative path.

| Relative path | Contents | Retention/authority |
| --- | --- | --- |
| `data/state.db` | SQLite scheduler, query, candidate, status, provenance, and deduplication state | Canonical mutable operational database; back up before schema or policy changes |
| `data/state.db-wal`, `data/state.db-shm` | SQLite write-ahead-log sidecars when WAL is active | Operational sidecars; never copy or delete independently while the DB is active |
| `data/raw_video/` | Newly downloaded source media awaiting routing, plus retained raw sources required by current policy | Mutable staging/source area; misses may be purged according to `loop.purge_raw_on_miss` |
| `data/transcripts/{uid}.json` | WhisperX transcript, segments, and available timing/alignment for each processed source | Canonical transcript corpus; intended to be retained even when raw misses are purged |
| `data/hits/{uid}/` | Witnessed positive-source directories | Canonical retained witnessed media |
| `data/hits/{uid}/metadata.json` | Witnessed reactions, scene/provenance, source metadata, and clip references | Canonical witnessed metadata |
| `data/hits/{uid}/clip_*.mp4` | Derived witnessed event/reaction clips | Retained derived media; may have multiple reaction records per clip |
| `data/hits/{uid}/reaction_*.txt` | Human-readable reaction text artifacts where produced | Derived provenance/inspection artifact |
| `data/instructional/{uid}/` | Instructional source record and demonstrations | Canonical retained instructional media |
| `data/instructional/{uid}/metadata.json` | Instructional norms, demos, quotes, bounds, polarity, genre, and provenance | Canonical instructional metadata |
| `data/instructional/{uid}/demo_*.mp4` | Derived instructional demo clips | Retained derived media; exact visual alignment remains audit-gated |
| `data/discussion/{uid}.json` | Commentary/normative-statement records | Canonical commentary text-label corpus |
| `data/discussion_video/{uid}.{ext}` | Retained full commentary source videos | Canonical commentary source-video storage on sk3 |
| `data/discussion_video/.stage/` | Atomic download staging for commentary backfill | Temporary; should contain only in-progress artifacts |
| `data/discussion_video/backfill.jsonl` | Dailymotion/Rumble/general commentary backfill attempt ledger | Append-only operational provenance |
| `data/discussion_video/youtube_backfill_sk3.jsonl` | sk3 YouTube worker download/failed-attempt ledger | Append-only worker provenance |
| `data/discussion_video/youtube_backfill_sk3.lock` | sk3 YouTube worker lock | Ephemeral coordination file |
| `data/negatives/{uid}/` | Same-source no-reaction, no-violation, and null-verified negative clips/metadata | Canonical negative-control corpus |
| `data/audio_events/{uid}.json` | Full PANNs/AudioSet event timeline and peaks for witnessed sources | Derived append/resume scoring output; ranking evidence, not truth |
| `data/audio_recover_journal.jsonl` | Resumable nonverbal-reaction recovery decisions | Append-only recovery journal |
| `data/metadata/` | Auxiliary per-source metadata configured by `paths.metadata` | Operational metadata; inspect schema before use because pillar metadata also lives beside media |
| `data/shadow_scores/` | Corpus-wide append-only feature, VLM, manifest, queue, and weak-supervision outputs | Canonical shadow-scoring area; does not mutate source corpus labels |
| `data/visual_audit/` | Compact visual-audit database and derived audit summaries | Canonical audit-ledger area when present on sk3 |
| `data/quarantine/<reason>/` | Reversible historical cleanup/quarantine, including duplicates, title filters, commentary reroutes, hard negatives, and pre-recut copies | Preserve until explicit reviewed disposition; not a trash directory |
| `data/hits_instr_quarantine/{uid}/` | Witnessed directories quarantined as likely instructional contamination | Reversible staged media |
| `data/deployment_backups/<timestamp>_<change>/` | Pre-deployment code/config/state artifacts created by deployment scripts | Rollback evidence; retention should follow an explicit policy |
| `data/monitor/hourly_collection/` | Append-only hourly health snapshots, latest snapshot, review queue, and monitor lock | Observational monitoring output |
| `data/retro_journal.jsonl` | Retroactive witnessed cleanup actions and retry state | Append-only operational journal |
| `data/qa_sample.jsonl` | Exported QA sample when the spot-check sampler is used | Derived review input |
| `data/qa_journal.jsonl` | Already reviewed QA items, preventing repeated samples | Append-only review journal |
| `data/channel_block_log.jsonl` | Logged channel-policy blocks from the crawler | Append-only scheduling/provenance log |
| `data/splice_signals.jsonl` | Seeded/recorded leakage and splice-signal taxonomy instances | Research/provenance artifact |

### A.3 Strict-audit and full-corpus shadow locations

The latest monitor code names the following sk3 runs:

| Location under sk3 live root | Contents |
| --- | --- |
| `data/shadow_scores/20260811_strict_audit_backfill_v1c/` | Latest final strict-audit coverage/manifests and merged compact backfill state used by the hourly monitor |
| `data/shadow_scores/20260811_strict_audit_backfill_queues_v1/` | Strict-audit and low-level scoring queues/shards |
| `data/shadow_scores/20260806_witnessed_video_asr_v3/` | Witnessed video-frame plus ASR shadow inputs/outputs and retry manifest |
| `data/shadow_scores/20260724_full_corpus_v1/` | Earlier full-corpus shadow manifest/features used by later backfill builders |

An earlier report mentions
`data/shadow_scores/20260811_strict_audit_backfill_v1b/`; the final monitor and
calibration report use `v1c`. Treat `v1b` as an earlier checkpoint and `v1c` as
the latest frozen run name unless a new inventory says otherwise.

The compact local summaries corresponding to this remote state are:

- `audit_runs/20260811_strict_audit_backfill_v1/AUDIT_REPORT.md`;
- `audit_runs/20260811_strict_audit_backfill_v1/final_summary_20260811.json`;
- `audit_runs/20260811_strict_audit_calibration_v2/AUDIT_REPORT.md`.

### A.4 Local checkout contents

The local checkout does not contain the bulk video corpus. Its main data and
artifact locations are:

| Local path | Contents |
| --- | --- |
| `/Users/spangher/Projects/stanford-research/social-norms/src/` | Production pipeline source copied/developed locally |
| `/Users/spangher/Projects/stanford-research/social-norms/scripts/` | Audit, scoring, backfill, deployment, evaluation, and repair scripts |
| `/Users/spangher/Projects/stanford-research/social-norms/config/` | Runtime configuration, query plans, model profiles, weak-signal registry, and shadow policies |
| `/Users/spangher/Projects/stanford-research/social-norms/tests/` | Unit, contract, evaluator, selector, renderer, and deployment tests |
| `/Users/spangher/Projects/stanford-research/social-norms/docs/` | Research, audit, signal, operations, and roadmap documentation |
| `/Users/spangher/Projects/stanford-research/social-norms/audit_runs/` | Compact frozen audit manifests, ledgers, reports, selected proxies, model outputs, and exact repair artifacts synchronized into the checkout |
| `/Users/spangher/Projects/stanford-research/social-norms/data/visual_audit/audit.db` | Local compact SQLite audit ledger |
| `/Users/spangher/Projects/stanford-research/social-norms/data/visual_audit/audit.db-wal` | Local audit DB WAL sidecar when active |
| `/Users/spangher/Projects/stanford-research/social-norms/data/visual_audit/audit.db-shm` | Local audit DB shared-memory sidecar when active |
| `/Users/spangher/Projects/stanford-research/social-norms/data/visual_audit/search_source_audit_summary.json` | Compact source-audit summary |
| `/Users/spangher/Projects/stanford-research/social-norms/data/visual_audit/shadow_policy_v1_coverage.json` | Compact shadow-policy coverage summary |
| `/Users/spangher/Projects/stanford-research/social-norms/sample_clips/` | Small local sample media, not the canonical corpus |
| `/Users/spangher/Projects/stanford-research/social-norms/model_cache/` | Small/local model cache material, not the canonical shared server cache |
| `/Users/spangher/Projects/stanford-research/social-norms/patches/` | Captured live-file patches/snapshots and deployment support files |
| `/Users/spangher/Projects/stanford-research/social-norms/ops/` | Versioned copies of node-specific operational launchers |

When disk cleanup is needed, preserve compact ledgers, manifests, manual
judgments, reports, hashes, exact accepted artifacts, and provenance. Rendered
contact sheets, duplicate frame bundles, temporary encodes, and model caches can
often be regenerated, but should be removed only after their authoritative
ledger references are verified.

### A.5 Audit-run organization

All checked-in audit evidence is under:

`/Users/spangher/Projects/stanford-research/social-norms/audit_runs/`

The corresponding live/synchronized location on sk3 is generally:

`/lfs/skampere3/0/alexspan/norm-scraper/audit_runs/`

Audit directories are intentionally run-specific rather than one mutable
database. Common contents include:

- frozen source selection manifests;
- excluded prior-UID lists;
- transcript packets;
- storyboard/frame manifests;
- rendered contact sheets or proxy clips;
- raw VLM outputs;
- manually completed TSV/JSONL ledgers;
- post-reveal semantic adjudication;
- exact transformed MP4s;
- SHA-256 seals;
- evaluation JSON;
- summary JSON;
- an `AUDIT_REPORT.md` or `README.md`.

The 28-rule registry is the best index of which individual audit artifacts are
decision-authoritative:

- local: `config/audited_weak_signals_v1.json`;
- sk3: `/lfs/skampere3/0/alexspan/norm-scraper/config/audited_weak_signals_v1.json`.

The registry records exact repository-relative evidence paths and hashes. The
large historical audit chronology is in:

- `docs/weak_supervision_visual_audit.md`;
- `docs/audit_gated_rollout.md`;
- `docs/witnessed_reaction_signal_cycles.md`.

### A.6 Distributed YouTube recovery locations

The frozen launch evidence is local and, when synchronized, under the same
relative path on sk3:

`audit_runs/20260722_distributed_youtube_launch/`

Important files include:

- `README.md`: storage-integrity audit summary;
- `manual_review.jsonl`: nine manually reviewed worker outputs;
- `sheets/`: contact sheets when retained in the full audit bundle.

The frozen distributed manifest was created by
`scripts/build_distributed_youtube_manifest.py`. Its reported SHA-256 is
`8670ed9d5f6b70eb42d795d1ae088b93722922c703c8e0d2b09c6e21968984a3`.
The exact manifest filename/location is not stated in the checked-in launch
README; it should be added to the next on-host inventory.

Each worker receives an explicit `--storage-root`. It writes:

```text
<storage-root>/data/discussion_video/{uid}.{ext}
<storage-root>/data/discussion_video/.stage/
<storage-root>/data/discussion_video/youtube_backfill_<worker>.jsonl
<storage-root>/data/discussion_video/youtube_backfill_<worker>.lock
```

The sk3 canonical destination is the live repository’s
`data/discussion_video/`. The checked-in documentation says sk1 and sk2 outputs
remain node-local pending safe consolidation, but it does not record their exact
absolute `--storage-root` arguments. Therefore those two absolute paths are a
known inventory gap, not something to infer. Before consolidation, obtain them
with one batched read-only check and record them in a non-secret deployment
manifest.

### A.7 Logs, locks, and process state

| Location | Purpose |
| --- | --- |
| `logs/run_YYYY-MM-DD.log` under the sk3 live root | Main crawl stdout/stderr by launch date |
| `logs/crawl_watchdog.log` under the sk3 live root | Watchdog restart and duplicate-process warnings |
| `logs/retro.log` when used | Retro-cleanup output |
| `.batch_pipeline.lock` under the sk3 live root | Prevents overlapping deferred GPU batches |
| `/tmp/crawl_watchdog.lock` on sk3 | Prevents overlapping watchdog actions |
| `data/monitor/hourly_collection/monitor.lock` | Prevents overlapping hourly monitor snapshots |
| `data/monitor/hourly_collection/snapshots.jsonl` | Append-only hourly snapshots |
| `data/monitor/hourly_collection/latest.json` | Atomically replaced latest snapshot |
| `managed_servers/managed_vllm_events.jsonl` under the managed audit workspace | VLM wait/start/inference/unload lifecycle events |

The documented sk2 lifecycle ledger example is:

`/lfs/skampere2/0/alexspan/visual_audit/managed_servers/managed_vllm_events.jsonl`

Process IDs are ephemeral and should never be treated as durable state. Queue
and item status belong in `data/state.db` and the append-only ledgers.

### A.8 Python environments, ffmpeg, and model caches

#### sk3 runtime environments

| Location | Use |
| --- | --- |
| `/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/python` | Scraper, WhisperX, ffmpeg-adjacent pipeline, media/audit utilities |
| `/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/ffmpeg` | Canonical clip/render binary used by scripts |
| `/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/ffprobe` | Media validation |
| `/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python` | Offline vLLM detector and selected audit workloads |
| `/lfs/skampere3/0/alexspan/miniconda3/` | Historical/base conda installation |

#### Model caches

| Location | Use |
| --- | --- |
| `/lfs/skampere3/0/shared_hf_cache/` | Preferred shared Hugging Face model cache on sk3 |
| `/lfs/skampere3/0/alexspan/.cache/huggingface/` | User-specific fallback Hugging Face cache referenced by packaging scripts |
| `/lfs/skampere3/0/alexspan/.cache/torch/` | Torch checkpoint cache used by visual scoring |
| `/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362` | Configured offline 70B detector snapshot |
| `/lfs/skampere2/0/shared_hf_cache/hub` | Shared sk2 model cache used by managed Qwen lanes |

Model caches are regenerable infrastructure, not authoritative label data.
Avoid copying them into audit bundles or local disk unless required.

### A.9 Configuration and policy locations

| Path under repository root | Contents |
| --- | --- |
| `config/settings.yaml` | Runtime paths, scheduler, platform, batch, clipping, negatives, audio events, and model configuration |
| `config/search_terms.yaml` | Primary search seeds and sources |
| `config/instructional_terms.yaml` | Instructional query taxonomy |
| `config/reaction_phrases.yaml` | Reaction phrase tiers/tags |
| `config/norm_taxonomy.yaml` | Norm taxonomy |
| `config/typical_social_norm_queries_v1.yaml` | Ordinary social-norm query canaries |
| `config/audited_search_queries_v1.yaml` | Audited retrieval query set |
| `config/audited_search_query_maintenance_v2.yaml` | Reversible query maintenance policy |
| `config/audited_weak_signals_v1.json` | Governed signal registry and evidence hashes |
| `config/weak_supervision_shadow_v1.yaml` | Inactive evidence-backed shadow policy and historical audit observations |
| `config/sk2_audit_vllm_profiles.json` | Managed sk2 model profiles |
| `config/sk3_audit_vllm_profiles.json` | Managed sk3 audit-model profile |
| `config/channel_blocklist.json` | Channel exclusion policy |

### A.10 Secrets and credentials

Credential contents must never be copied into this document, audit JSONL,
logs, or source control.

Known credential classes are:

- three user-provided YouTube Netscape-cookie exports, installed separately on
  sk1, sk2, and sk3 with mode `0600`;
- the YouTube Data API key configured through
  `youtube.api_key_file`, currently `~/.youtube-data-api-key.txt` relative to
  the sk3 runtime home;
- the proxy list configured at
  `/lfs/skampere3/0/alexspan/.proxies-webshare.txt`;
- optional environment credentials such as `HF_TOKEN` or API keys used by
  explicitly launched audit clients.

The original local cookie exports were supplied from the user’s Downloads
directory, but those source files are not canonical runtime data and should not
be relied on as backups. The exact installed remote cookie filenames are not
checked into this repository. That omission is deliberate for secrecy, but a
secure private operations inventory should record host, owner, mode, expiry
check date, and a non-reversible fingerprint—not cookie contents.

### A.11 Ephemeral scratch and regenerable artifacts

Known scratch locations include:

- `/tmp/instr_frames/` on local and sk3 for the original 14-frame instructional
  toe-in-the-water audit;
- `.stage/` directories under download destinations;
- rendered frame/storyboard/contact-sheet directories inside individual
  `audit_runs/`;
- temporary exact-clip encodes before atomic move;
- model server logs and sockets inside managed audit workspaces;
- `.pytest_cache/` and Python `__pycache__/` directories;
- local `model_cache/` content.

Scratch is not authoritative unless an audit report explicitly seals it by
path and hash. Before deleting large audit artifacts, preserve:

- selection manifest;
- item IDs and source UIDs;
- exact temporal bounds;
- transform parameters;
- manual judgments and rationales;
- raw model outputs if they support an evaluation;
- SHA-256 hashes;
- evaluation and summary reports;
- exact accepted transformed clips that cannot be reproduced from retained
  source media.

### A.12 Known location gaps to close

The following are not fully recoverable from the checked-in documentation and
should be captured in a small, non-secret deployment manifest during the next
scheduled server inventory:

1. exact absolute `--storage-root` used by the sk1 distributed YouTube worker;
2. exact absolute `--storage-root` used by the sk2 distributed YouTube worker;
3. exact location of the frozen 1,128-record distributed YouTube manifest;
4. node-local completion/consolidation status for sk1 and sk2 commentary videos;
5. secure fingerprints—not contents—of the three installed cookie files and
   their last successful canary dates;
6. whether the sk3 `v1b` strict backfill directory is still retained after
   `v1c` became authoritative;
7. an explicit retention policy for `data/deployment_backups/`, old quarantine,
   rendered audit proxies, and model caches;
8. a current live inventory newer than the frozen 2026-08-11 snapshot.

Closing these gaps requires only one batched, read-only check per host. It does
not require walking the full media tree or repeatedly polling SSH.

---

## 21. Recommended immediate sequence of work

The recommended next execution order is:

1. **Freeze the revised target definitions.** Update schemas and audits so
   general witnessed validity no longer requires perfect independent-bystander
   identity, while preserving that subtype.
2. **Implement the standardized LF interface.** Start with the ten registry-
   approved mechanisms and the strongest atomic contradiction signals.
3. **Build LF diagnostics before Snorkel.** Measure coverage, overlap,
   conflicts, correlations, and audited precision/recall.
4. **Create larger source-disjoint gold cohorts.** Especially increase the
   number of genuine witnessed reaction positives and include broader responder
   roles.
5. **Fit pillar-specific Snorkel models in shadow.** Compare against individual
   signals and simple baselines.
6. **Audit the model outputs intensively.** Review all proposed positives and a
   random/low-score control sample.
7. **Backfill posterior scores over the existing corpus.** Keep every original
   video and store probabilities separately.
8. **Train a multi-head critic from the atomic manual labels.** Use it as a new
   LF and ranking model, not immediate truth.
9. **Continue query canaries and collection monitoring.** Judge retrieval by
   exact usable-event yield, not raw download counts.
10. **Promote only narrowly supported uses.** Retrieval, ranking, and rerouting
    can be promoted before automatic keep decisions if their audits support
    those limited purposes.

---

## 22. Final perspective

The project has accomplished two difficult things:

1. it has built a large, resumable, multi-platform corpus collection system
   with retained videos, transcripts, clips, negative controls, provenance, and
   operational safeguards; and
2. it has established an unusually strict audit culture that records failed
   mechanisms rather than quietly promoting them.

The current bottleneck is no longer raw scale. It is converting heterogeneous,
fallible evidence into calibrated judgments about whether a concrete social
event is actually present and usable in a bounded audiovisual interval.

The strongest path forward is not one more monolithic VLM question. It is:

- define the targets precisely;
- decompose them into observable atoms;
- preserve abstention;
- group correlated evidence;
- combine signals probabilistically;
- validate on source-disjoint human gold;
- use open models for scalable ranking;
- use closed-model/manual review for calibration and hard cases;
- preserve all media and all provenance;
- and promote only the uses that fresh audits genuinely support.

That approach makes the existing 10k–50k-scale pillar collections useful even
before perfect labels exist. It supports multiple dataset views, enables
iterative improvement, and prevents early filtering mistakes from permanently
destroying valuable source material.
