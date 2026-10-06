# To-dos — approach AFTER dataset collection is complete

These are the analysis + research items deferred until the corpus is collected. They all concern one underlying question: **why do some norm violations draw a bystander reaction while others don't — and can we perceive/articulate the difference?**

The framing that organizes everything below: a bystander reaction is a **costly social act**, so the witnessed signal is a **non-random filter** on the universe of norm violations. A model trained on `data/hits/` learns "what gets reacted to," not "what is a violation." Characterizing that selection function is both (a) the core dataset-bias audit and (b) a genuine contribution on informal social control. Same question.

---

## The selection function (conceptual dimensions to test)

A reaction occurs only when perception + standing + acceptable cost cross threshold, amplified by severity, possibly dampened by crowd. Dimensions:

**A. Perception (was it seen as a violation, in the moment)**
- Conspicuity / visibility (overt vs covert/later-apparent)
- Ambiguity / pluralistic ignorance (clear violation vs "accident or shove?")

**B. Standing (is it the bystander's place; are they able)**
- Concrete victim present (defendable party) vs diffuse/absent victim
- License / role relationship (peer, in-group, role-authority vs stranger)
- Power asymmetry / retaliation risk (violator higher-status, bigger, armed, boss)

**C. Valence (is it worth the cost)**
- Severity / moral vs conventional (Turiel)
- Descriptive-norm alignment (widely flouted → nobody reacts even if injunctively forbidden)

**Cross-cutting**
- Crowd size / diffusion of responsibility (can suppress; or amplify cheap expressive signals)
- Reaction modality (verbal confrontation vs glare/withdrawal/deferred-report) — **our detector only captures verbal/audible disapproval**, so "no reaction" often means "no audible reaction"

---

## To-dos

### A. Measure the selection function on the current corpus (analysis — run on existing data)
- [ ] Aggregate `severity × reaction_strength × n_people × reactor_role × violator_role × scene_type` across `data/hits/` (`provenance.scene`). Report whether reaction_strength tracks severity, whether high-n_people crowds suppress, and the reactor_role mix (victim vs bystander vs authority).
- [ ] Within-norm contrast: find norm categories appearing in **both witnessed and commentary**, and compare what differs (setting, crowd, victim presence). Commentary = violation that drew NO in-situ reaction, so the witnessed-vs-commentary split is itself a reaction/no-reaction partition for the same violation type.
- [ ] Map the **injunctive/descriptive gap**: norms heavily represented in `instructional` (taught) but rare in `witnessed` (drifted/descriptively-permitted) vs the reverse. Quantify per category.
- [ ] Profile the `drop`/`miss` population (norm-ish content, no detected reaction) — is it "no violation" or "violation, no reaction"? Needs sampling/QA to be usable as a control.

### B. Matched counterfactual — same/nearly-same scenes WITHOUT a reaction  *(user priority)*
Goal: for witnessed violations (got a reaction), find near-twin scenes of the **same** violation that got **no** reaction, match on observables so the reaction is the only varying factor, then identify what discriminates. This is the empirical test of the framework above — it isolates *what actually causes* the reaction.
- [ ] Define the matching key: same norm category + similar scene (`scene_type`, `n_people`, setting) + similar severity. Match treated (witnessed, reacted) to controls (same violation, no reaction) via nearest-neighbor on scene features.
- [ ] Identify control sources for **"violation-but-no-reaction"** (distinct from our "no-violation" nulls):
  - candid `drop`/`miss` rows with norm content but no bystander response
  - commentary videos of the same norm (no in-situ reaction)
  - within-video: same video, similar act elsewhere that drew no reaction
  - manual/VLM-verified curation of violation clips confirmed to have no reaction
- [ ] On the matched pairs, measure the discriminating features (conspicuity, victim presence, violator power/role, crowd size, ambiguity) → that delta is "what is different about violations that get reacted to."
- [ ] **Open challenge to flag:** sourcing verified violation-but-no-reaction controls is hard because the detector is reaction-mediated (it finds violations *through* reactions). Likely needs a dedicated collection pass or a VLM-verified manual set. Scope this when we get there.

### C. Dataset-balancing implication
- [ ] Use B's results to decide whether/how to **balance** the dataset with violation-no-reaction controls, so the model learns "what is a violation" rather than "what gets reacted to." This is a new negative/counterfactual class distinct from the no-violation nulls in `data/negatives/`.

### D. Open conceptual questions (worth articulating as we go)
- [ ] How much of "no reaction" is *modality* (silent/deferred response we can't hear) vs *abstention* (noticed, disapproved, chose not to act)? These look identical in transcript-only data.
- [ ] Is the reaction signal stable across cultures/languages, or does standing/severity weightings shift? (We have multilingual data to test.)
- [ ] Does the selection function differ by norm *type* (etiquette vs safety vs harassment)? Hypothesis: harassment reactions are gated heavily by power asymmetry; etiquette reactions by crowd/setting.

---

## Why this is the core question, not a side note
The bystander reaction oversamples visible, victim-present, severe, unambiguous, low-cost-to-react, verbally-reactable violations, and undersamples covert, diffuse-victim, ambiguous, powerful-violator, silently-responded-to ones. Measuring and (via B) counterfactually testing this filter is the difference between a model of *norm violations* and a model of *what bystanders happen to confront*.
