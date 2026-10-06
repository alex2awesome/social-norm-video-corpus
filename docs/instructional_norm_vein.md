# Instructional / demonstrated-norm video vein

**Date:** 2026-06-03
**Idea (user):** beyond candid violation+reaction clips, scrape **didactic** videos that
*teach* social norms — autism/SEL social-skills lessons, corporate harassment/DEI
training, "social stories" — where a role-play **DEMO** enacts a norm
violation/situation and an **educator narrates the norm**. The narration is
essentially **free ground-truth annotation**.

## Why this is a real contribution (the gap)

No existing dataset captures **didactic social-norm video**. Prior work splits two ways:
- **Candid/scripted social-norm video — norms are *implicit*, require inference + paid annotation:** EgoNormia (1,853 egocentric MCQs, 7 norm cats; arXiv 2502.20490), VideoNorms (~1k clip-norm pairs from 8 TV shows; 2510.08543), Social-IQ / Social-IQ 2.0, Social Genome (272 videos, 1,486 reasoning traces with *evidence spans* — closest analog to "explanation = ground truth"; 2502.15109), SIV-Bench.
- **Instructional-video understanding — but for *physical* skills:** COIN (11,827 how-to videos w/ step boundaries; 1903.02874), HowTo100M (100M narrated how-to clips; 1906.03327). No *social-skills* equivalent of COIN exists.

**Our novelty:** a corpus where the norm is (a) **explicitly stated by an educator** and
(b) **enacted in a labeled role-play demo** → high-precision, *self-labeled* ground truth
(the narrator's own words), which the candid datasets must hand-annotate. Self-labeled
VideoQA items ("withhold the explanation → name the violation") are a uniquely strong artifact.
(Reaction-as-signal precedents: BAD bystander-affect 2303.04835; the candid pipeline we already run.)

## Two genres

### A. Autism / SEL social-skills
Highest annotation density = formats that pair a demo with explicit "why":
- **Everyday Speech** (everydayspeech.com) — student-actor role-plays + thought-bubbles/voiceover explaining the *why*. **Cleanest match.** Free samples: pages.everydayspeech.com/scc-samples.
- **Social Thinking / "expected vs unexpected behavior"** (Michelle Garcia Winner, socialthinking.com) — the vocabulary itself ("that was unexpected because…") is a labeled contrast. *Older* videos richer (framework since softened).
- **Carol Gray Social Stories** (carolgraysocialstories.com) — narrated rule statements; video versions on YouTube/TPT (often demo-light).
- **Howard B. Wigglebottom** (We Do Listen Fdn) — animated, explicit moral. **Sesame Street / Julia** ("See Amazing"; free). **Model Me Kids** (real-kid modeling; demo-heavy, narration lighter).
- Other sources: **Teachers Pay Teachers** ("video social story", "expected vs unexpected behavior video"), **Vimeo** (Be Like Buddy: vimeo.com/belikebuddy), Move This World, Centervention, Org. for Autism Research / SSM Treffert Center free libraries.
- Subreddits (search per-sub, weak Google indexing): **r/slp, r/specialed, r/ABA, r/AutismParenting, r/autism, r/socialskills** (+ r/Teachers, r/ECEProfessionals).
- *Caveats from research:* "Watch Me Learn" real but verify URL on YouTube; **"MarapodAU" not found — drop.**

### B. Corporate / workplace training
Best fit (demo + explicit verbal naming of the violation): **sexual-harassment, bystander-intervention, DEI/microaggression**.
- Providers (gated full courses, free samples/trailers): **Traliant** (+ acquired Kantola, Emtrain), **Media Partners** ("Once & For All" — flagship; "Dealing With Conflict"), **Atana** (free clips, "Unintentional Still Hurts", "you are so well spoken"), **Skillsoft** (free YT sample youtube.com/watch?v=irfYrvBDQFc), **ELI** (Civil Treatment), **Second City Works** ("Real Biz Shorts", "Your Best Defense"), EVERFI/ComplyEQ, Litmos, LinkedIn Learning.
- **Gold for clean DEMO+NARR: classic VHS uploads on YouTube** — "Kmart Sexual Harassment Training 1994" (WDR6fws2XYE), "…1989" (gXjhLnwYjas), IUNXPFI6dYU, umpPCCSnZ0g. Also EEOC video page; TikTok parodies.
- Subreddits: **r/humanresources, r/AskHR, r/sysadmin** (mandatory-training memes), r/antiwork, r/corporate, r/instructionaldesign (format experts).
- *Caveat:* title "Sexual Harassment: It's Not Enough to Know Better" attributed to Media Partners is **unconfirmed** — verified flagship is "Once & For All."

## Detection signal — the canonical unit
**NORM-STATEMENT → DEMO (named-character dialogue) → DEBRIEF (labeling + "because" + corrected replay).**
The debrief sentence with a *labeling verb + violation noun* ("this **is** quid pro quo / a microaggression / unexpected") is the highest-precision anchor; the preceding named-character dialogue is the demo it annotates.

Segment register: NARRATION/EXPLANATION (educator, 2nd-person, modal/imperative, meta-words "norm/appropriate/should") vs ROLE-PLAY DEMO (in-character, names, situated/quoted speech). Cues ranked:
1. **Diarization shift** (WhisperX speaker IDs — turn on `transcribe.diarize`): narrator = 1 consistent speaker; demo = 2+ alternating. Fails when narrator also acts.
2. **Discourse markers (high precision):** open = "let's watch / here's an example / in this scenario / imagine / now the wrong way"; close = "did you see that? / cut / now let's discuss / what went wrong".
3. **Person/tense/modal shift** (cheap POS feature).
4. **Pause gaps + music stings** (from WhisperX word timestamps) + **PySceneDetect shot boundaries** (talking-head vs scene = different camera).
5. **Visual-narration alignment** (WYS², arXiv 2301.02307) — speech depicted vs talking-head.
Hard cases: narrator-also-acts; cold-open demos w/ no marker; nested good-vs-bad; **silent visual-only demos** (transcript-blind — needs video model); ironic quoted narration.

## LLM detector (separate prompt; `is_instructional` short-circuits to candid pipeline if false)
Invert the candid signal: here the **explanation IS the label** (extract verbatim, don't infer), and the demo is a *deliberately staged* violation. Extract: `is_instructional`(+conf), `genre`, `norms[]` (canonical statement + category), `demo_scenarios[]` (polarity violation|correct|contrast_pair, start/end verbatim quotes + timestamps + speaker ids + open/close markers), `explanation{quote,time,refers_to}` (**ground truth**), `expected_behavior{quote,statement}`. Keep field names parallel to the candid detector so both feed one schema.

## Clipping strategy (differs from candid 20s-pre-reaction)
Boundaries are **content-defined**, not fixed windows. Per video save:
1. **Demo clip** = `[demo.start,end]` (word-timestamps + ~1s shot pad) — the enacted norm event, precisely bounded.
2. **Explanation clip** = the debrief span, stored as the **paired label** (the value-add: no human annotation).
3. **Pre-explanation demo** (demo *without* the following explanation) = a self-labeled "name the violation / what's the norm?" VideoQA item, gold answer withheld. **Strongest research artifact.**
4. **Contrast pairs** (bad-then-good of same norm) linked by `norm_id` + polarity — "which is correct / what changed" eval, impossible from candid video.
5. Norm-statement intro clip as context (keep separable so it can be withheld at eval — else it leaks the answer).

## Sources / terms (seed sets)
~40 autism/SEL terms (social story X, expected vs unexpected behavior X, social skills role play X, video modeling X, whole body listening, size of the problem…) and ~40 corporate terms (sexual harassment training role play, bystander intervention 5 D's, microaggression training scenario, "is this harassment" vignette, 1990s harassment training cringe, Media Partners "Once and For All"…) are in the research (full lists in agent reports). To implement: add an `instructional` taxonomy block + run on YouTube (Data API) + Vimeo + TPT + the listed subreddits; route via a new `is_instructional` detector branch.

## Status / next
Research only — **not yet implemented.** Build order when greenlit: (1) `instructional` search-term set + seed on YouTube/Vimeo/subreddits; (2) `is_instructional` LLM detector branch (above schema) routing to a separate `data/instructional/` corpus; (3) demo/explanation clipping; (4) turn on WhisperX diarization to aid demo segmentation.
