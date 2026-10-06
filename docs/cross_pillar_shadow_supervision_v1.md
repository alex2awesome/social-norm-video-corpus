# Cross-pillar atomic shadow supervision v1

The implementation is `scripts/cross_pillar_shadow_supervision_v1.py`. It is a
shadow review scheduler, not an automatic corpus filter. Every output fixes
`automatic_acceptance=false`, `corpus_disposition=null`, and
`delete_media=false`.

The input fields are questions with enumerable answers
(`yes|no|uncertain`), not free-standing weak labels. A reviewer or model records
what can actually be observed; code groups correlated observations into one
label-function family. Missing evidence abstains.

Witnessed asks whether there is a human social event, a distinct on-scene
non-authority bystander, a response targeted at that event after it occurs, an
observable reaction, an organic source, and a clean pre-reaction training
span. Instructional asks whether a human social event is concretely
demonstrated, actor/action/target belong to the same event, the extracted norm
and polarity match it, and the demonstration can be cut away from the
explanation. Commentary separately asks whether the transcript grounds a
specific occurred event and stance, whether that event is actually visible,
whether it is temporally localized, whether the repaired actor-action-target
label matches, and whether label-bearing editorial text is absent.

The family—not the individual prompt—is the aggregation unit. For example,
four social-scope atoms yield one `social_scope` vote. This prevents correlated
paraphrases from overwhelming independent evidence in a future Snorkel label
model.

The initial prospective audit is frozen at
`audit_runs/20260807_cross_pillar_supervision_v1/`: 60 recent source-disjoint
videos, 20 per pillar, with three frames per source and a complete manual
ledger. Confirmed visual target rates were 15% instructional, 0% strict
witnessed, and 35% commentary. Sparse positives were deliberately left with
unreviewed atoms as `uncertain`; materializing them produced no automatic or
complete candidate. Named negative families may rank likely reroutes for
review, but no new keep rule passed this audit.

The automatic query scope gate is separate. It was audited on 90 manually
judged proposals (54 ordinary social queries and 36 drift/adversarial queries)
with zero false allows and zero false blocks in that frozen cohort. It affects
only `llm_expand`, `exploit`, and `explore`; curated instructional, commentary,
taxonomy, and negative queries are untouched. Existing rows are reversibly
marked `policy_excluded`, never deleted, and every future proposal is appended
to `query_proposals` with its policy version and decision.
