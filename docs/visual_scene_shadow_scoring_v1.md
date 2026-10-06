# Visual scene shadow scoring v1

Status: calibration contract only. Scores are append-only evidence. They do not
delete, quarantine, relabel, recut, or reroute a corpus item.

## Why the score is factored

One binary `is_good` label hides different failure modes. Every candidate must
retain independent axes:

1. `text_social_norm`: does the proposed text resolve to an actor, concrete
   action/situated speech act, and affected person or genuinely social shared
   setting?
2. `situated_scene`: does the video show characters acting inside a scenario,
   rather than only a presenter, commentator, graphic, or context B-roll?
3. `screen_or_graphic_demo`: does a screen recording, interface, diagram,
   animation, or other non-situated medium visually demonstrate the labeled
   behavior or outcome without relying on narration alone?
4. `label_alignment`: does the visible scenario plausibly demonstrate the
   proposed norm, rather than merely sharing its topic?
5. `event_completeness`: is the decisive action/exchange complete, partial,
   context-only, or described-only?
6. `authenticity`: organic, staged, mixed, or unresolved. This affects
   witnessed routing, not instructional usefulness.
7. `label_leak`: do captions, narration, a reaction, or graphics explicitly
   supply the answer?
8. `repairability`: can a bounded temporal cut, crop, mute, or concrete relabel
   produce a usable artifact?

## Operational questions

`text_social_norm=yes` requires all of:

- a named or context-resolved actor;
- a recognizable action, omission, or situated speech act;
- another person or social commons affected by the conduct;
- an informal expectation that observers can approve or disapprove.

Technical procedures, equipment/appliance operation, exercise, health/self-care,
and abstract values without a resolved act remain separate. Vague labels such
as `respect`, `equality`, and `inclusion` are repair candidates when the context
supports a concrete relabel; they are not silently promoted.

`situated_scene=yes` includes:

- staged role-play, scripted film/TV excerpts, hidden-camera scenes, organic
  recordings, animation, and screen-mediated scenarios;
- single-actor conduct in a public/shared context;
- dialogue and other situated speech acts between characters.

It excludes audience-directed talking heads, interviews that only report an
event, news B-roll/aftermath, and label-bearing graphics without the act.

`screen_or_graphic_demo=yes` is separate from `situated_scene`. It accepts a
visually self-contained interface action, animation, or diagram when the
medium itself demonstrates the proposed behavior or outcome. It excludes a
topic slide, decorative B-roll, or captions that merely describe an off-screen
event. This branch preserves valid non-photorealistic demonstrations without
weakening the human-scene requirement.

## Pillar-specific interpretation

### Instructional

Any demonstration medium is allowed. Organicity is not required.

Current audited score bands:

- `instructional_high_confidence`: Qwen and GLM both say situated demo, and the
  Qwen3-32B operational text gate says social norm.
- `instructional_likely_review`: either video VLM says situated demo, and the
  text gate says social norm.
- `instructional_repair`: a scene is visible but the label is vague/off-target,
  the interval is too long, or label-bearing narration/captions need removal.
- `instructional_low_priority`: neither visual nor text evidence passes.

On a new 50-source holdout, the high-confidence band had 0.900 precision and
0.643 recall; the likely-review band had 0.706 precision and 0.857 recall.
These are ranking bands, not destructive rules.

### Witnessed

The violation action must be visible before and independently of the reaction.
Organic authenticity and causal reaction targeting must be scored separately.

Open VLMs do not currently pass this gate. On 57 manually judged sources split
into pre-reaction microclips, the best tested combination had only 0.400
precision and 0.600 recall. Witnessed VLM outputs are therefore restricted to:

- candidate ranking for manual/closed-model audit;
- detecting likely scripted scenes for instructional rerouting;
- proposing a pre-reaction interval for review.

No witnessed auto-accept band exists in v1.

### Commentary

The accepted text statement remains useful even when the video is commentary
only. Visual recovery is a separate localization task:

- find a cutaway, embedded source clip, reenactment, screen scenario, or
  on-camera speech act that depicts the discussed event;
- verify that actor/action/target identity survives without relying on
  descriptive captions or commentator narration;
- record whether mute/crop/temporal-cut sanitization is needed.

On 13 source-disjoint manually reviewed targets, dual-VLM whole-target
agreement reached 0.778 balanced accuracy, 0.667 precision, and 0.667 recall.
After splitting the same targets into 119 overlapping six-second windows,
requiring both models to see a situated scene in at least three windows reached
0.875 balanced accuracy, 1.000 precision, and 0.750 recall. The missed positive
was a valid on-screen tipping-interface demonstration, which motivates the
separate `screen_or_graphic_demo` branch. A five-source replication was perfect
for GLM and had one Qwen context-only false positive. Both samples remain too
small for promotion.

## Scalable feature stack

All features are computed from immutable proxies and stored beside source
hashes and model/rubric versions:

- motion, histogram change, shot-change rate, and face counts;
- Keypoint R-CNN person/pose counts;
- image CLIP depiction-type prompts;
- temporal X-CLIP scene/direct-address prompts;
- transcript discourse and off-screen-description priors;
- Qwen3-32B operational social-norm text result;
- Qwen3-VL-8B and GLM-4.6V-Flash structured video judgments;
- per-window evidence for overlapping six-second microclips.

Motion, pose, CLIP, X-CLIP, and transcript scores schedule expensive review.
They never independently certify a datapoint.

## Append-only record

Each shadow record should contain:

```json
{
  "item_id": "pillar:uid:index",
  "source_sha256": "...",
  "source_mtime": 0,
  "proxy": {
    "sha256": "...",
    "fps": 3,
    "max_width": 640,
    "duration_sec": 0
  },
  "text_gate": {
    "model": "Qwen3-32B",
    "rubric": "v2",
    "result": {}
  },
  "cheap_visual": {
    "motion": {},
    "pose": {},
    "clip": {},
    "xclip": {}
  },
  "visual_axes": {
    "situated_scene": null,
    "screen_or_graphic_demo": null,
    "label_alignment": null,
    "caption_dependent": null,
    "aftermath_or_context_only": null
  },
  "video_vlm": {
    "rubric": "v3",
    "qwen": {},
    "glm": {},
    "windows": []
  },
  "shadow_band": "high_confidence|likely_review|repair|low_priority",
  "scored_at": 0
}
```

Records live outside corpus metadata. A later policy may consume them only
after a new source-disjoint audit reproduces its intended precision/recall
target.

## Execution constraints

- Use bounded pre-rendered proxies. Qwen processor-side resampling has an
  audited off-by-one failure; long GLM inputs exceed context.
- Whole clips longer than the context budget enter the repair/localization
  band. Do not use `any positive window` as automatic acceptance: long talk
  shows often contain unrelated social scenes.
- Use overlapping six-second windows with a three-second stride only for
  uncertain/long/brief-action candidates.
- Preserve every failed model response and HTTP/processor error.
- Never write a shadow score into source `metadata.json`.
