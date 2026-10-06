# Weak supervision and post-training plan v1

## Principle

Do not ask one model whether a clip "contains a social norm." Decompose each
pillar into observable atomic claims, keep every asset, and predict one of
`candidate`, `review`, or a named reroute/rejection reason. Every operating
threshold must be selected on a frozen manual holdout and verified again on a
fresh prospective sample.

The 2026-08-07 recovery batch provides immediate hard negatives: accident
compilations, police/authority footage, animal incidents, gameplay, and
off-screen news descriptions were sometimes promoted because reaction language
alone was present. A deterministic 12-source visual sample found 1/4 clear
instructional demonstrations, 0/4 clips satisfying the strict witnessed
bystander contract, and 3/4 commentary sources with a visually present event
(one staged). This is a diagnostic sample, not a population estimate.

## Shared atomic signals

1. **Human-social event**: a human action or communicative act affects another
   person or shared social setting. Separate this from accidents, animal
   behavior, gameplay, sports, police procedure, and generic spectacle.
2. **Event observability**: the asserted action is visible/audible in the saved
   interval, rather than described off-screen or represented by a portrait,
   slide, or unrelated B-roll.
3. **Semantic alignment**: actor, behavior, affected party, norm, and polarity
   agree across transcript span, title/context, and video window.
4. **Temporal alignment**: the event occurs before/around the weak signal, with
   a measurable start/end and enough unlabelled visual evidence to train on.
5. **Source/format provenance**: organic, staged demonstration, animation,
   news package, compilation, gameplay, police/authority, and talking-head are
   tags, not a universal keep/drop decision. Their meaning is pillar-specific.
6. **Novelty and cluster contamination**: perceptual hashes, transcript
   embeddings, channel, and related-query ancestry prevent one bad series from
   dominating a split or an audit.

Cheap metadata/ASR rules should retrieve or veto only unambiguous families.
Pose, person tracks, speaker embeddings, audio events, and VLM judgments should
be stored as independent signals with provenance and confidence; none is an
automatic label until audited.

## Witnessed: model the response, not generic norm recognition

The strict target is a causal sequence:

1. a human event occurs;
2. a distinct non-authority witness is present;
3. that witness plausibly perceives the event;
4. the witness responds after it through speech, gesture, facial expression,
   approach/withdrawal, help, or intervention; and
5. the training action interval can be separated from the response signal.

Useful labeling functions:

- human/person tracks plus stable identities across event and response;
- diarization/speaker embeddings showing the reactor is not the violator;
- second-person disapproval, imperatives, defense/help language, gasp/laughter,
  prosodic surprise, and turn-taking after the event;
- head/body turn, gaze proxy, approach, pointing, shielding/helping, and abrupt
  pose/motion change by a previously uninvolved person;
- event-before-response ordering and a clean pre-response splice;
- negative LFs for first-person victim reaction, authority response, narrator
  commentary, crowd ambience, accident/spectacle, animals, gameplay, and
  produced compilations.

The current dense-video plus real-audio VLM result (75% precision / 50% recall
on its audited cohort) is useful as one LF, not a final classifier. A separate
role-binding head should answer `violator`, `target`, `bystander`, or
`authority/unclear` per tracked person.

## Instructional: require an aligned demonstration

Any format is allowed, including staged scenes, animation, and speech acts, if
the saved interval actually demonstrates the social behavior. Atomic checks:

- transcript grounds a specific actor, behavior, affected person/shared
  context, and normative obligation;
- behavior is literal social conduct, not a technical procedure, metaphor,
  personal optimization, or abstract policy claim;
- a demonstration segment exists and is temporally distinct from explanation;
- pixels/audio in that segment instantiate the grounded behavior;
- the proposed polarity (`violation`/`correct`) matches what is demonstrated;
- repeated/redundant demos from one source are clustered rather than counted as
  independent evidence.

High-value negatives from the recovery sample include NFC "smart labels"
misread as etiquette and a digital-marketing talking head retrieved by a
workplace-role-play query. Text semantic atoms should catch the domain error;
video/text alignment should catch the absent demo.

## Commentary: separate textual labels from visual supervision

Commentary remains useful text supervision even when the event is off-screen.
For visual learning, add three separate outputs:

- `concrete_event_statement`: the speaker refers to a specific event, not a
  general opinion;
- `occurred_in_source`: the referenced event is actually contained in the
  retained source video;
- `localized_visual_event`: a bounded interval visibly/audibly supports the
  statement.

Use transcript quote times, deictic phrases ("look", "here", "this person"),
scene changes, OCR/caption timing, replay/B-roll boundaries, motion/person
tracks, and a VLM verifier on short candidate windows. News portraits and
slideshows should remain text labels but receive `visual_event=no`.

## Combining weak labels

A Snorkel-style label model is appropriate only after measuring LF coverage,
precision, and correlation on frozen manual gold. Use named atomic LFs rather
than multiple paraphrases of the same LLM prompt. Fit separate label models per
pillar and output calibrated posterior bands:

- high-confidence candidate;
- disagreement/manual-review priority;
- named likely failure/reroute.

Do not train/test split by clip alone: group by source video, channel, perceptual
cluster, and query ancestry. Otherwise duplicate series make performance look
artificially strong.

## Post-training options

1. **Multi-head quality critic**: train a small video/audio/text ranker on the
   frozen manual ledgers to predict the atomic claims above, not the final norm
   category. Distill closed-VLM/manual judgments into an open model only after
   agreement is measured on unseen source clusters.
2. **Pairwise ranking**: teach the critic that an audited good interval should
   rank above a nearby talking-head, off-screen, accident, or wrong-role window
   from the same source. Within-source pairs reduce genre shortcuts.
3. **Positive-unlabeled learning**: treat strict manually approved examples as
   positives and the rest as unlabeled, with explicit high-confidence hard
   negatives. Do not assume every unapproved corpus item is negative.
4. **Cross-modal disagreement mining**: prioritize cases where ASR says a
   social event occurred but vision sees only narration, or vision sees an event
   but the extracted norm/polarity disagrees. These cases are maximally useful
   for the next audit round.
5. **Training-dynamics triage**: after a warm-up model, audit persistent
   high-loss, high-uncertainty, and high-influence examples. Use these signals
   to schedule review, never as an unaudited deletion rule.
6. **Teacher ensemble and calibration**: combine transcript LLM, video+audio
   VLM, role/pose/audio models, and retrieval provenance; calibrate confidence
   per pillar with source-clustered cross-validation and conformal or isotonic
   bands.

## Audit loop

For each proposed signal: preregister its exact computation and threshold;
freeze a source-disjoint manual set; report coverage, precision, recall, Wilson
intervals, and failure strata; run a fresh prospective replication; then deploy
in shadow. Only after both audits pass may it change a keep/reroute decision.
All original media and raw signal outputs remain recoverable.
