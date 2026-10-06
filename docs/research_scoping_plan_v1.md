# Research scoping plan v1 (2026-09-03)

Five research programs on top of the weak-supervision layer, a common
experiment infrastructure to scope them, and honest notes on the confounds
the user flagged. Everything here is shadow/append-only and consumes the
existing ledgers (posteriors, tags, source-context labels, cut manifests).

## The five programs and their scoping pilots

### R1. Reaction prediction (forward model of informal social control)
**Task:** given a pre-reaction clip, predict reaction presence, responder
role, strength (1–5), and latency. **Negative controls (scoped 2026-09-03):**
19,966 negative sources / 81,706 clips exist; **2,912 sources (27% of
positives) have both an action clip and a same-source negative window** →
matched within-video pairs that hold channel, style, and scene constant.
Remaining controls: unmatched no-reaction windows and null-verified
mundane-twin sources. Known asymmetry to audit: negatives are
detector-clean windows, so they may skew calm; the matched subset is the
defensible core. **Pilot:** manifest of positives (organic ∪ genuine-reaction
staged, social norm domains) vs matched negatives; zero-shot VLM baseline +
linear probe on frozen features; report AUROC + calibration by tier/platform.

### S1. The selection function (what gets reacted to)
Observational estimates of reaction strength/presence against severity,
head-count, norm domain, content type, platform; witnessed-vs-commentary
composition per norm domain as the reaction/no-reaction partition. Pure
analysis over existing ledgers (no GPU). Confound register: severity and
strength are detector-assigned (calibrate against gold), platform
concentration (85% Dailymotion), query-lineage selection.

### E1. Enculturation channels / norm induction
**Theory grounding (user asked):** no single theory names exactly these
three channels; the trichotomy is our synthesis of established constructs —
direct instruction ≈ natural pedagogy (Csibra & Gergely) and teaching in
cultural-transmission models (Boyd & Richerson; Henrich); observed sanction ≈
observational/vicarious learning (Bandura) and third-party norm enforcement
(Tomasello's developmental work); commentary ≈ gossip/reputation as norm
transmission (Dunbar; Feinberg & Willer). The mapping is an empirical bet,
not settled theory — which is fine, because the ablation tests it.
**Channel confounds (user's point, correct):** instructional scenes are
cleaner, often text-overlaid, staged for legibility; commentary carries
narration. Controls: leakage probes as standing metrics (OCR-text-only
probe, audio-mute ablation, caption-region masking test), difficulty
matching on cheap visual stats (motion, person count, resolution), and
within-norm-category matching across pillars. **Pilot:** for the top ~10
norm categories present in all three pillars, norm-articulation eval — model
sees k clips from one channel, states the rule, scored against instructional
statements (the 5.5k explanation-only YT transcripts are the rule bank).

### A1. Norm-aware agents in simulation
Reaction predictor → social-cost model → CARLA/highway-env agent. Scoped
experiments: censure/task-reward exchange rate; audience-sensitivity probe
(internalized norm vs surveillance sensitivity — behavior with vs without
bystanders present); over-conservatism check. Interface contract now, sim
work after R1 exists: `cost(scene_description | clip) -> p(reaction),
E[strength]`.

### C1. Counterfactual deltas
Observational: matched pairs (R1 controls + commentary parallels + screened
purged-miss "violation, no reaction" candidates). Causal: sim sweeps varying
only the action. Cheap middle rung: text/generative minimal edits scored by
the reaction predictor. Depends on R1; no separate infra.

## Common infrastructure (build order)

1. **Task manifests** (`experiments/` package): join ledgers → per-task item
   lists with media paths, labels, covariates, and source-disjoint splits
   (group by uid; channel/cluster grouping when available). One schema for
   all tasks. [BUILT: reaction-prediction manifest v1]
2. **Analysis module**: selection-function crosstabs/regressions over
   manifests. [BUILT: v1 first pass]
3. **VLM baseline runner**: frames-in, JSON-out batch runner mirroring
   `run_source_context_llm_v1` (same dynamic load/unload discipline), Qwen-VL
   class models from the existing sk2/sk3 profiles; atomic question sets per
   task, never holistic norm questions. [NEXT]
4. **Leakage probe suite**: text-only, audio-only, OCR-only probes runnable
   against any manifest; leakage reported beside every headline metric. [NEXT]
5. **Eval/report module**: AUROC, PR, calibration (reuse label-model
   calibration_report), per-slice breakdowns (tier, platform, language,
   content type), Wilson intervals. [reuses weaksup pieces]

## Sequencing
R1 manifest + S1 analysis now → VLM zero-shot baselines on R1 (small pilot,
~500 items) → leakage probes → E1 pilot → gold wave doubles as human
baseline for R1 → sim interface once R1 metrics exist.
