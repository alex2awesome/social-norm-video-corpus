# Audited weak-signal registry v1

This registry turns the tested search, text, visual, and multimodal signals into
one fail-closed operational contract. It is defined in
`config/audited_weak_signals_v1.json` and enforced by
`scripts/weak_signal_registry.py`.

Every evidence artifact is repository-relative and SHA-256 sealed. Validation
checks source-disjoint manual coverage, metric arithmetic, artifact hashes, and
the declared use. A failed-transfer rule must abstain. Unknown experimental
signals are reported rather than silently consumed.

| Rule | Pillar | Audited use | Key held-out evidence |
| --- | --- | --- | --- |
| Audited scene queries | all | candidate retrieval | 60 full sources manually reviewed; query text is never a label |
| Dailymotion related parent gate V1 | witnessed | **disabled after transfer; related queue quarantined** | Development parent audit was 20/20 decisions correct, but a source-disjoint child transfer found 0/20 confirmed strict witnessed clips in the proposed allow band (19 reject, 1 uncertain); a 20-child blocked comparison also had 0 confirmed. All 10,735 queued IDs and all media are preserved. |
| Clip-wide intervention scan | witnessed | reaction-window proposals | 6/6 strict reactions found, with 12 false proposals: 33.3% precision, 100% recall |
| Qwen silent-video + ASR reaction retrieval V3 | witnessed | **disabled after corpus transfer** | a fresh unbiased 60-source audit produced 1 TP, 6 FP, and 2 FN (14.3% precision, 33.3% recall) |
| Exact-span authority cue | witnessed | exclude from strict bystander route; preserve visible-scene review | discovery transfer 9 TP, 0 FP, 6 FN; independent confirmation 24 TP, 0 FP |
| Creator/staging title cue | witnessed | exclude from organic route; send to instructional review | 32/32 precise first audit; refined independent audit 38/38 |
| Identity/multimodal ensemble | witnessed | **disabled** | failed source-disjoint transfer |
| V20 multimodal band | instructional | **disabled** | failed fresh transfer; high and low sampled bands both contained 20% visual demos |
| V23 Qwen/Gemma consensus | instructional | **disabled after replication** | Initial 36-source result was 4 TP/1 FP, but a preregistered 100-source replication produced 6 TP/8 FP/10 FN: 42.9% precision, 37.5% recall; all 400 model outputs manually reviewed |
| Scene-oriented title cues | instructional | retrieval and review ranking only | 30 TP, 6 FP, 66 FN across 185 audited sources: 83.3% precision, 31.3% recall for finding a visual demo |
| Non-explanation polarity | instructional | manual-review ordering only; explanation clips are preserved | two independent 100-demo audits captured 56/60 visual demos (93.3% recall); precision varied from 26.0% to 50.7%, so this is not a keep label |
| Capture/title event cue | commentary | localization candidate and ranking | 93/154 usable sources; 74/154 exact-title events |
| Dual-VLM retrieval core | commentary | high-priority localization review | development: 42 TP, 4 FP, 51 FN (91.3% precision, 45.2% recall); source-disjoint transfer: 7 TP, 0 FP, 11 FN (100% precision, 38.9% recall), with all 46 model outputs manually reviewed |
| Dual-VLM bounded clip proposal | commentary | **disabled for automatic keeping** | every one of 32 transformed clips received blind artifact review plus post-reveal semantic review; only 2/32 passed every action, bounds, audio, label-leakage, and alignment check |
| Adaptive edge-text crop | commentary | **disabled** | on all 11 known exact-event/text-leaking parents, guarded OCR geometry abstained on 9; both rendered artifacts failed manual review (one news tail, one persistent editorial target circle) |
| Raw commentary statement as visual action label | commentary | **disabled** | complete source-disjoint manual text review accepted 0/30 extracted phrases unchanged; 16 sources remained eligible for visual search only after actor–action–target relabeling |

## Shadow LF votes (added 2026-08-17)

A sixth declared use, `shadow_lf_vote`, lets an audited mechanism cast one
`+1`/`-1` vote for the generative shadow label model
(`scripts/label_model_v1.py`). It is registered per rule through a
`shadow_lf_vote_contract` naming exactly one revised-ontology target, one
signed vote, one evidence family, the audited precision/recall the model may
use (null when an audit found the metric unstable), and `shadow_only: true`.
Validation rejects a contract without the declared use, a declared use without
a contract, and any contract on a failed-transfer rule. A shadow vote affects
no retrieval, route, acceptance, or media — it is combined, in shadow, by the
label model and calibrated against human gold before any further promotion.

Five mechanisms currently hold the use, all on the strength of their existing
frozen audits:

| Rule | Vote | Target | Basis |
| --- | --- | --- | --- |
| Clip-wide intervention scan | +1 | `reaction_grounded` | 33.3% precision / 100% recall on the source-disjoint 24-source scan |
| Exact-span authority cue | -1 | `independent_bystander_signal` | pooled v2+v3 audits: 33 TP, 0 FP, 6 FN |
| Creator/staging title cue | -1 | `independent_bystander_signal` | pooled audits: 70 TP, 0 FP, 11 FN |
| Official WWYD channel | -1 | `independent_bystander_signal` | 36 TP, 0 FP, 6 FN |
| Non-explanation polarity | +1 | `demonstration_present` | 93.3% recall pooled; precision unstable (0.26–0.51) and therefore recorded null |

The three exclusion cues vote only against the strict organic bystander
subtype, never against the broad `norm_event_supported` target. Retrieval and
ranking rules without a contract continue to abstain: query or title text is
never a label.

There are deliberately zero acceptance-gate rules. Combined records contain
review routes, strict-route exclusions, and ranking evidence, while
`acceptance_label` and `corpus_disposition` remain null and `delete_media`
remains false. This is not a claim that no signal can ever become an acceptance
gate; it means none has yet passed the independent promotion contract.

The Dailymotion recommendation graph is now independently disabled from Reddit
subreddit discovery. The parent-side V1 rule appeared perfect on its 20-parent
development cohort (one true propagation parent and 19 true blocks), but failed
the outcome that matters: its historical allowed parents produced no confirmed
strict witnessed clip in a deterministic 20-child transfer sample. The queued
`dmrelated` rows are marked `policy_excluded` with
`related_transfer_failed:20260808_v1`; they were not deleted or deactivated and
can be restored exactly. The transfer also identified staged/prank/film items
worth separate instructional review, but that salvage does not justify routing
the graph into witnessed.

`scripts/combine_audited_shadow_scores.py` is the append-only bridge from score
JSONL files to this contract. It only consumes fields named by exact registry
rule IDs; unknown fields cannot trigger routes, and failed-transfer rules
abstain even when an old score file contains a positive value.

The latest V23 replication gate is
`audit_runs/20260806_instructional_v23_replication_100/maintenance_gate_result.json`.

Search retrieval has a separate reversible maintenance contract in
`config/audited_search_query_maintenance_v2.yaml`. A complete 128-unit blind
and post-reveal audit found zero end-to-end strict yield for six of the eleven
originally activated exact queries. The maintenance action only sets those
exact query rows inactive; it does not delete queries or videos, alter labels,
or reactivate positive-yield queries that another process had already paused.

The two active instructional rankings were also evaluated jointly on two
source-disjoint manual cohorts (136 demos). Retro scene-scan provenance was
fully nested inside non-explanation polarity, so the two signals must not be
treated as independent Snorkel votes. They do support a replicated review
ordering: both signals were 16/22 visual demos (72.7%), polarity alone was
14/85 (16.5%), and neither was 2/29 (6.9%). Corpus scoring placed 2,055 of
58,707 demos in the highest tier, 27,490 in the middle tier, and preserved
29,162 in the lower tier. No automatic disposition was created.

The commentary dual-model rule passed its preregistered independent transfer
gate. On 23 rendered sources, strict Qwen3-VL-8B/Gemma-3-27B agreement
selected seven and all seven were manually confirmed (95% Wilson precision
interval 64.6%--100%). It remains a review-ordering signal because it missed
11 of 18 usable sources. Manual review of every model output found the main
errors to be missed visible events, plus isolated actor/target reversal,
invented action, and bad temporal bounds.

High source-ranking precision does not transfer directly to exact clips. A
separate exhaustive audit of all 32 frozen bounded proposals found visible
performed events in 16, but only two artifacts passed the complete clip
contract. Label-bearing news text appeared in 23, and wrong or ambiguous
temporal localization was also common. The exact-clip rule is therefore
registered as failed-transfer; only the two individually audited artifacts
carry manual acceptance, and no automatic corpus disposition is authorized.

The subsequent adaptive edge-text crop also failed its frozen mechanism gate.
It safely abstained when OCR entered the central event region or an OCR-free
crop would be too small, but this left only two renderable outputs. Manual
review rejected both: one retained a long unrelated news transition and one
retained a red dotted target marker. This confirms that generic cropping is
not a reliable remedy for burned-in editorial packaging.

A new source-disjoint commentary text audit also disabled the raw detected
statement phrase as a visual action label. All 30 selected statements were
reviewed; 16 sources had a sufficiently grounded event and stance to justify
full-source visual search, but every one required replacing the detected phrase
with a concrete actor–action–target description. The common error was selecting
the reaction, consequence, apology, justification, or abstract judgment as if
it were the depicted act. This failed rule does not reject the source video: it
only prevents the unreviewed phrase from becoming supervision. The 16 corrected
labels are now in a separate visual localization audit and still carry no keep
decision.
