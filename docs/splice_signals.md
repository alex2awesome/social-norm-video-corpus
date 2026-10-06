# Weak-supervisory splice signals — tracking leakage from the bystander reaction

Goal (2026-07-13): during QA audits, record which observable signals let us (a)
**localize the norm violation** and (b) **cut it cleanly so the bystander reaction
doesn't leak** into the violation clip / label. This is the groundwork for the
deferred "non-leaking datapoint" translation ([[commentary-vein]],
[[instructional-vein-dormant]]).

Findings live in `data/splice_signals.jsonl` (one row per reviewed uid).

## The leakage structure (from `clip_extract.py` + 11-sample audit)

`extract_clips` cuts each witnessed clip as `[reaction_start − pre, reaction_start + post]`
(defaults pre=12s, post=2s). Two structural facts:

1. **The +2s post-roll already contains the reaction onset.** A non-leaking
   violation clip should end at `reaction_start`, not `reaction_start + 2s`.
2. **Reactions within `merge_window` (22s) are bundled into one clip**, so a clip
   often spans a whole exchange with violation+reaction interleaved.

### Four regimes (classify every witnessed datapoint into one)

| Regime | Timestamped utterance | Leakage risk | Splice strategy |
|---|---|---|---|
| `verbal_violation` | The violating utterance itself (insult / threat / inappropriate question) | **Low** — no separate reaction to leak | `utterance_is_label` — the utterance IS the datapoint |
| `physical_violation_reaction` | Bystander's verbal response to a visible/physical violation | **Real** — reaction at clip end; violation loose in pre-roll | `cut_at_react_start` — keep pre-roll, cut at `reaction_start`; VLM for exact violation frame |
| `mixed_exchange` | Alternating violation/reaction turns in one bundled clip | **High** — reaction spread throughout | `needs_turn_segmentation` — need speaker diarization |
| `narrating_witness` | Reactor narrates the violation as it happens | Medium | `cut_at_react_start` |

11-sample distribution: verbal_violation 27%, physical_violation_reaction 27%,
mixed_exchange 36%, narrating_witness 9%.

## Signals tracked per datapoint (`splice_signals.jsonl` schema)

```
uid, pillar, assessed_at, assessor,
regime                         # verbal_violation | physical_violation_reaction | mixed_exchange | narrating_witness
best_splice_strategy           # utterance_is_label | cut_at_react_start | needs_turn_segmentation | unspliceable
splice_signals:
  react_verbal_boundary        # clean | fuzzy | n/a      (sharp onset of bystander verbal reaction at reaction.start)
  violation_locatability       # tight (utterance==violation) | loose (in pre-roll) | interleaved | none
  audio_event_at_reaction      # high | low | none         (commotion/scream peak co-occurs — catches physical, MISSES verbal)
  speaker_diarized             # bool                      (reactor vs violator turn boundary — currently null everywhere = GAP)
  multi_reaction_bundled       # bool                      (clip spans >1 reaction within merge_window)
  clip_leaks_reaction_onset    # bool                      (post-roll includes reaction.start — structurally true today)
  n_reactions, clip_span_sec
vlm_needed[]                   # exact_violation_frame_in_preroll | visual_reaction_in_preroll_frames | speaker_turns
note
```

## Key gaps the audit is surfacing (candidates for later work)

- **Speaker diarization is null everywhere.** Populating `speaker` (WhisperX
  already runs in `batch_transcribe`) would make `mixed_exchange` splicable and
  sharpen every boundary. Highest-leverage fix.
- **audio_events misses verbal arguments** (returns 0 for heated speech). Useful
  only for physical commotion; don't rely on it as a universal boundary.
- **Violation is only loosely localized in Regime B** (somewhere in the 12s
  pre-roll). Tightening needs a VLM pass (the earlier codex:rescue VLM vetting
  is the right tool) — flag `vlm_needed: exact_violation_frame_in_preroll`.

## How to apply during audits

For each reviewed witnessed uid, add one `splice_signals.jsonl` row (regime +
strategy + the signal fields above). Aggregate per-regime every cycle so we know
the corpus's spliceability distribution before designing datapoints. Educational
and commentary rows differ (see below) and are sparser — mostly note their
structurally-different leakage mode.

### Educational (`data/instructional`) — leakage is structurally absent
`demos[]` already carry `start`/`end` + `start_quote`/`end_quote`/`explanation` →
the violation/explanation is **tightly timestamped**, and there is **no bystander
reaction to leak**. These are the cleanest datapoints. Row fields: regime=
`verbal_violation`, violation_locatability=`tight`, clip_leaks_reaction_onset=
`false`. Main risk is *test-set leakage between modalities* (same norm appears in
witnessed training) — track overlap separately.

### Commentary (`data/discussion`) — text-only, different leakage mode
No video clip (`save_discussion` writes text only). Leakage concern is inverted:
the narrator's `statements[]` *state the norm explicitly*, so if those quotes are
fed as input the label leaks. Datapoint translation here likely means using the
quote as the **label/target**, not the input. Row fields: regime=`commentary`,
vlm_needed=[], note the quote-as-label framing.
