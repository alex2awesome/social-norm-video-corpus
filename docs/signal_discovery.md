# Norm-violation signal discovery (transcript analysis, 2026-06-02)

Four subagents independently read 23 transcripts (~9.8k words; Reddit + Dailymotion
hits *and* misses) hunting for norm-violation signals our keyword detector misses.
Strong cross-agent convergence — the families below were each found independently
by ≥2 agents.

## New signal families (ranked by convergence + value)

| # | Signal | Found by | Example (uid) | Suggested pattern |
|---|--------|----------|---------------|-------------------|
| 1 | **Appeal to authority** (cops/911/security/manager/corporate/lawsuit) | all 4 | "Come on, get security." / "I'll call the cops." (x6flsy2); "I will sue the fuck out of you." (1tu70eh) | `\b(call(ing)?\|get\|ring\|have)\s+(the\s+)?(cops?\|police\|911\|security\|manager\|corporate)\b`; "press charges", "I'll report you", "sue you" |
| 2 | **Command to leave / stop / back off** | all 4 | "Get off my mom's property." / "Walk away. Walk away." (x6flsy2); "Get out of the store!" (x6d8uil) | `\b(get out\|get off (me\|my\|us)\|walk away\|back off\|stay away\|leave (my\|the))\b` |
| 3 | **Threat to escalate (violence)** | all 4 | "I'll punch you right in your fucking face." (x6flsy2); "My husband will kill you." (x6d8uil) | `\b(i('?m gonna\|'?ll\|will)\s+(kick\|punch\|beat\|fuck .* up\|crack)\b)`; "what you gonna do" |
| 4 | **Property / law / rule citation** | A,C,D | "Read your lease." / "arrested for trespassing." (x6flsy2); "It's private property." / "First Amendment right." (x6d8uil) | "trespass(ing)", "private property", "read your lease", "the contract", "you have no right", "First Amendment", "public forum" |
| 5 | **Recording-as-sanction / go viral** | all 4 | "I just recorded this." / "Where are you going to post it? YouTube?" (x6flsy2); "Thanks for letting me go viral." (1tu70eh) | `\b(record(ing)?\|filming\|on camera\|go(ing)? viral\|post (it\|this)\|take your picture)\b` |
| 6 | **Directed dehumanizing insult** ("you're a ___") | all 4 | "you're a wanker" (x6d8uil); "take your fucking medication" / "I hate fat pig men" (x6flsy2) | `\byou('re\| are)? (a\|an )?(wanker\|perv\|creep\|liar\|sick\|crazy\|insane\|pathetic\|dumbass\|moron)\b`; "take your medication" |
| 7 | **Violator entitlement / credential defense** | all 4 | "My dad's the owner, so good luck with that." (x6flsy2); "I know the rules. I know what you're allowed to do." (x9icbse); "I have a right to be here." (x6d8uil) | "my dad('s) the owner", "I know the rules", "I have a right to", "do your job", "good luck with that" |
| 8 | **Conditional ultimatum** ("if you ___ again, I'll ___" + "do you understand?") | A,C | "You honk at me again, I'm going to kick your ass… you understand?" (x6flsy2); "Next time you fucking touch me…" (1tudsi3) | `\bif you\s+\w+.*(I'?ll\|I('?m)? gonna\|or I)\b`; "do you understand" (repeated) |
| 9 | **Coerced apology / pleading (victim distress)** | B,C | "No, I'm sorry. Please let me go." (1tudsi3); "I'll stop. I'll stop." (1tudsi3) | repetition gate: ≥3× "please"/"I'm sorry"/"let me go" in a short span |
| 10 | **Third-party bystander narration** | A,B,C | "Can someone stop this guy?" (x3xi3db); "oh my gosh you saw him just throw the trash can" (1tued2k) | `\b(can\|someone\|somebody) .{0,10}stop (this\|him\|her)\b`; "this (guy\|dude\|lady)" + complaint verb |
| 11 | **Demand to identify / accusation of lying** | C | "What is your name, sir?… You told me you was the owner." / "You are a liar." (x7td68b) | "what'?s your name", `\byou('re\| are)? (a )?li(ar\|ed)\b`, "changing your story", "that's not what you said" |
| 12 | **"Mind your business" / boundary-policing** | A | "Mind your business." (x6flsy2); "nothing to do with this" | `\bmind your (own )?business\b`, "leave me alone", "stay out of it" |
| 13 | **Daring physical contact** | B,C | "Touch me. Touch me." (1tu70eh); "I want you to hit me." (x7td68b) | `\b(touch\|hit) me\b` (esp. repeated), "put your hands on me" |
| 14 | **Repeated single-word aggression (chanting)** | A | "Bitch! Bitch! …" (~16×); "Mind your business" ×3 (x6flsy2) | `\b(\w+)([!.,\s]+\1\b){2,}` (token repeated 3+×) |
| 15 | **Sarcastic / hostile politeness** | all 4 | "Have a nice day, ma'am." (repeated, x6flsy2); "Forgive me. Forgive me." (x6d8uil) | **weak feature only** — gate on nearby insult/threat; never standalone |

## Critical methodological findings

1. **These signals add real RECALL — proven.** Two of the richest confrontation
   transcripts (`dailymotion__x7td68b` Subway/HOA disputes; `dailymotion__x6d8uil`
   trespass/contract standoffs) are labeled **is_hit=0** — our current detector
   *missed* them entirely. They are full of signals 1–7. → the phrase list is too
   narrow; profanity + direct-censure alone misses authority/property/command-driven
   confrontations.

2. **Song lyrics pollute the corpus.** Lewis Capaldi's "Someone You Loved" appeared
   in **5 files** (x9u3lf0, x9utriy, x9viyzy, x9wa294, x8rjzcq) — music videos
   mis-surfaced by search, and their "so let me apologize / please" language can
   false-trigger censure detectors. → **add a lyrics/music gate** (e.g. drop videos
   whose transcript is highly repetitive or matches known lyrics; or filter by
   category/title before transcription).

3. **Hard negatives need filtering.** Police-procedural arrest audio
   (reddit__1tu648x — UK caution), sports broadcast (reddit__1tu7132 — baseball
   ejection), and a **staged prank** (x5vlb5r — fake knife threat, "Prank is going
   on… we are making a video") all look superficially confrontational. The prank
   case is especially important: staged content is an adversarial false positive.

4. **Near-duplicates.** x3xi3db ≈ x4dicbc (same compilation, minor ASR diffs) —
   dedupe to avoid double-counting.

5. **Modality split.** x9icbse is podcast hosts *recounting* freakouts (reported
   speech, not live confrontation) — a different signal class ("I'm sitting there
   witnessing… a whole group of us watching this lady freak out").

6. **Precision still needs gating.** Many signals (commands, "touch me", sarcastic
   politeness, "what's your name") are high-recall but fire in benign contexts —
   they should be **co-occurrence features** (count multiple signals within a window)
   rather than standalone hits. This is the recall/precision tradeoff to tune, and
   the spot where the LLM-as-noisy-filter (not judge) belongs.

## Recommended next steps

- Add signals 1–7 as new high-value tiers in `reaction_phrases.yaml` (authority,
  command, threat, property/law, recording, directed-insult, entitlement).
- Add 9, 11, 13 as medium tiers; 8, 10, 12, 14 as features; 15 as a gated weak feature.
- Add a **music/lyrics filter** and a **staged-prank / hard-negative filter**.
- Move from "≥1 phrase = hit" toward a **score = weighted sum of signals in a window**,
  so authority+command+threat co-firing ranks above a lone profanity token.
