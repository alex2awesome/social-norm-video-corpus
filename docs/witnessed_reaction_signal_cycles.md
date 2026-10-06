# Witnessed reaction signal cycles (2026-07-29)

## Target

These experiments intentionally separate reaction detection from the downstream
witnessed acceptance contract. The target is:

> A distinct third party gives a contemporaneous, targeted response to another
> person's behavior.

Authenticity, tacit-norm validity, action visibility, exact semantic label, and
clean pre-reaction splicing are separate downstream labels.

## Literature-derived mechanisms

The BAD dataset paper (`arXiv:2303.04835`) treats reaction as a temporal change
from a person's neutral state. It tests manual reaction intervals,
failure-versus-control labels, and before-versus-after-failure labels. The
useful design lesson is the within-person temporal contrast, not its reported
precision: BAD uses controlled webcam views of a single known bystander, unlike
our edited, multi-person web video.

Wang and Yang's *That's So Annoying!!!* paper uses `#petpeeve` as a noisy
behavioral retrieval marker, then finds that lexical, dependency, and semantic
frame features improve categorization. For our reaction transcript, the
analogous units are the reaction predicate, addressee/target, intervention
type, and the preceding behavior span. The paper is
`https://cs.stanford.edu/people/diyiy/docs/emnlp_wang_2015.pdf`.

The bystander-intervention literature commonly separates direct intervention,
distraction, delegation, victim support, and delayed follow-up. Our current
strict reaction lexicon over-focuses on direct commands and should represent
these other mechanisms explicitly.

## Cycles and results

The initial 80-item manual audit contains 44 apparent reaction positives when
the explicitly adjudicated “not targeted third-party response” failures are
treated as negatives.

1. Metadata, reaction text, two open-VLM atomic judgments, and whole-scene
   audio deltas were combined with source-grouped out-of-fold logistic and
   random-forest models. The best apparent operating point was 88.9% precision
   and 54.5% recall.
2. Error audit found a cohort confound: all 22 rows missing legacy detector
   metadata were in the apparent positive complement. On the 58
   metadata-complete rows, the best combined result fell to 53.8% precision and
   63.6% recall; average precision was 0.53.
3. BAD-inspired reaction-centered features were added: before/after audio
   energy and onset strength, visual motion delta, motion peak at onset,
   histogram-cut magnitude, face-count delta, and largest-face motion delta.
   They did not transfer. The temporal-only model had 33.3% precision and
   22.7% recall; the all-feature model had 52.0% precision and 59.1% recall.

The exact report is
`audit_runs/20260729_witnessed_reaction_feature_cycles_v1/report_v3_temporal.json`.
All results are shadow-only and no corpus file was changed.

## Interpretation

Whole-frame and whole-audio temporal deltas are the wrong approximation to the
BAD setup. Edits, camera motion, music, and speech from the actor swamp the
bystander signal. A scalable next mechanism must first establish a reactor
track or speaker turn, then measure change on that identity:

- speaker change immediately after the alleged action;
- reaction utterance predicate and explicit target/addressee;
- face/head/gaze change on a person who was already present but not acting;
- multiple-bystander synchrony;
- separation of direct target response from a third-party response;
- causal ordering and a short response latency;
- edited-scene/cut boundary as an abstention signal;
- direct, distract, delegate, support, and delayed-intervention subtypes.

Snorkel-style aggregation should be revisited only after these labeling
functions have distinct error mechanisms. Combining several correlated
transcript/VLM variants did not recover the label in the current audit.

## Identity and source-disjoint cycle

The proposed identity-conditioned mechanisms were implemented in
`scripts/witnessed_reaction_identity_features.py`:

- exact speaker-turn change when diarization labels exist, with explicit
  abstention when they do not;
- separate, explicitly named acoustic speaker-change proxies (MFCC, pitch,
  spectral, and energy change), never presented as diarization;
- transcript response latency and preceding-turn structure;
- direct, distract, delegate, support, evaluation, warning, delayed-follow-up,
  self-defense, reported-speech, and generic-affect mechanisms;
- reactor/actor role separation from provenance and VLM atomic votes;
- cross-boundary face tracks, head/gaze/expression change, and multi-person
  synchrony;
- scene-cut abstention.

All feature primitives and evaluators have unit coverage. The earlier
metadata-complete 58-item audit did not improve: the all-feature logistic model
reached 54.5% precision and 54.5% recall, while the identity-only model reached
28.6% precision and 36.4% recall. The local transcripts had no diarization
speaker IDs, so exact speaker-change features correctly abstained throughout.
The report is
`audit_runs/20260729_witnessed_reaction_feature_cycles_v1/report_v4_identity.json`.

A zero-UID-overlap 24-video audit was then used as a promotion gate. It has six
strict bystander reactions, or ten when authority/host interventions are
reported as third-party reactions. Frozen selected-boundary rules failed:

- strict two-of-three VLM/text vote: 25.0% precision, 16.7% recall;
- best strict individual VLM: 50.0% precision, 33.3% recall;
- prior-58 trained full feature model: 30.8% precision, 66.7% recall;
- extended authority-inclusive dual-VLM intersection: 71.4% precision, 50.0%
  recall.

The report is
`audit_runs/20260729_witnessed_reaction_feature_cycles_v1/source_disjoint_24_v1.json`.
Nothing passed the acceptance-label promotion gate.

## Clip-wide localization cycle and corpus expansion

Manual error inspection showed that the detector-selected quote is frequently
not the reaction that makes the clip useful. It may be the actor's excuse,
later appreciation, or a generic line even when an earlier bystander
intervention is present. Therefore
`scripts/witnessed_reaction_candidate_scan.py` scans every transcript segment
and proposes bounded windows around explicit intervention mechanisms.

On the 24-video source-disjoint audit, this proposal stage recovered all six
strict bystander-reaction clips (100% recall) but proposed 12 additional
negative clips (33.3% precision). It is therefore approved only as a candidate
generator, never as an acceptance LF.

The audited proposal mechanism was expanded non-destructively with
`scripts/score_witnessed_reaction_candidate_windows.py`. On sk3 it scanned
9,653 current witnessed metadata directories and 20,903 clips, with zero
metadata failures. It emitted 21,881 candidate windows across 10,323 clips:

- direct: 19,502;
- evaluation: 797;
- distract: 573;
- support: 473;
- delegate: 350;
- warning: 299.

The frozen artifact is
`audit_runs/20260729_witnessed_reaction_candidate_expansion_v1/candidates.jsonl`
on sk3. Every row has `acceptance_label: null`, `corpus_disposition: null`, and
`policy: proposal_only_shadow_non_destructive`.

## Localized VLM reranking cycle

Forty-five candidate windows from the 24-video audit were rendered as dense
storyboards. Candidate VLM V1 supplied the transcript phrase and allowed
audible grounding. Manual error inspection invalidated this rubric: a silent
storyboard cannot identify an off-screen speaker, and Qwen inferred speaker
roles from the supplied words (for example, treating “Don't cry” as a distinct
bystander). GLM V1 was cancelled before loading.

V2 explicitly forbids audio and speaker inference from text and asks only for
a visibly distinct responder. Qwen completed all 45 windows and unloaded under
the managed GPU lifecycle. The strict all-atom rule predicted zero positives
because the model set temporal order to `no` in all 45 rows, including rows
whose own evidence described visible intervention. A diagnostic conjunction
that leaves visual order unresolved reached 60.0% precision and 50.0% recall
on strict bystanders (three true positives, two false positives). The two false
positives were precisely the unresolved identity cases: an involved
altercation participant and an affected target.

The Qwen V2 report is
`audit_runs/20260729_witnessed_reaction_feature_cycles_v1/candidate_vlm_qwen_visual_v2_eval_fixed.json`.
It is not approved for corpus gating. GLM V2 subsequently completed all 45
windows and unloaded under the managed GPU lifecycle. Its strict visual-only
rule also predicted zero positives: 41 windows were assigned no responder role
and four were assigned an actor role. It therefore recovered none of the six
strict manual positives and is likewise not approved for corpus gating.

The present supported architecture is therefore:

1. clip-wide transcript mechanisms for high-recall localization;
2. explicit diarization when available, otherwise abstain;
3. visual-only responder evidence as a separate shadow signal;
4. no Snorkel promotion until a new untouched audit shows that role identity
   and temporal order transfer.

## ECAPA acoustic speaker-proxy cycle (2026-08-05)

An additional identity experiment used SpeechBrain's VoxCeleb ECAPA encoder on
the retained source audio for all 45 transcript-localized candidates in the
24-item source-disjoint holdout. Candidate voice embeddings were compared with
up to four transcript turns before and after the candidate. The emitted fields
are explicitly named `speaker_proxy.*`; this is not diarization and does not
identify a person.

All 45 candidates decoded and scored, and all 45 corresponding storyboards
were manually reviewed. The complete ledger is
`audit_runs/20260729_witnessed_reaction_feature_cycles_v1/speaker_embedding_proxy_v1_manual_storyboard_audit.tsv`.
The acoustic change rule reached 100% recall but only 35.3% precision for the
six strict bystander clips (6 true positives, 11 false positives). It fired on
ordinary dialogue turns, actors' excuses, directly affected targets, and
authority/host speech. Intersecting it with the Qwen visual candidate rule did
not alter Qwen's selection at all: precision remained 60% and recall 50%.
The stricter “third voice sandwiched between the same speaker” pattern selected
zero clips.

Decision: the ECAPA feature failed as an identity or acceptance filter. It is
not registered or expanded to the corpus. The failure reinforces that
`new voice != bystander`: voice change can aid temporal proposal generation,
but role identity still requires situated visual or audiovisual grounding.
No media was moved, deleted, accepted, or rejected. The exact report is
`audit_runs/20260729_witnessed_reaction_feature_cycles_v1/speaker_embedding_proxy_v1_summary.json`.

## V6 input-modality correction (2026-08-06)

The frozen V6 role/causal-binding prompt incorrectly told Qwen3-VL-8B that it
could use the original audio inside the submitted MP4. Qwen3-VL consumes text,
images, and video frames; it is not an audio-capable Omni model. Supplying a
`video_url` therefore did not make the audio track available to the model. The
separately supplied ASR remained available, but ASR cannot identify which
visible person spoke.

V6 is consequently retained only for complete manual error analysis and is
hard-failed in its promotion evaluator. Every output still receives the
promised manual structured-decision and rationale review; any claim to hear
audio is marked unsupported. The new V7 prompt names the real input as ordered
video frames plus ASR and requires `offscreen_or_unresolved` whenever visual
speaker/role binding is absent. A genuinely audiovisual labeling function must
use an audio-capable model under a new transfer audit.

The sealed correction is in
`audit_runs/20260806_witnessed_v6_audio_capability_correction/`.
