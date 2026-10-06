# Weak-Supervision LF + Label-Model Implementation (v1)

**Date:** 2026-08-17
**Implements:** roadmap `docs/project_goals_progress_and_remaining_work_20260817.md`
sections 12 and 16 (Priorities 1–3) and section 21 steps 1–3 and 5.
**Policy:** everything below is shadow-only and append-only. No module inspects
media, creates an acceptance label, or deletes anything.

---

## What was implemented

### Milestone A — unified target ontology (`scripts/target_ontology_v1.py`)

- One versioned schema (`target_ontology_v1`) freezing the revised targets of
  roadmap section 2.3 (`norm_event_supported`,
  `event_audiovisually_grounded`, `label_alignment`, `reaction_grounded`,
  `independent_bystander_signal`, `clean_training_span`) and the per-pillar
  label-model target lists of section 12.4.
- Witnessed validity is generalized: `derive_norm_event_supported` accepts a
  normative signal from **any** responder role.
  `derive_independent_bystander_signal` is preserved as a strict subtype whose
  `no` never counts against the broad target.
- Grounding modes distinguish `physical_action_visible`,
  `situated_speech_audiovisual` (both ground), `description_only` and
  `absent` (never ground), `uncertain` (abstain).
- Repairable labels route to `repairable_relabel_review`
  (`label_disposition`) instead of being folded into reject.
- All composites are deterministic tri-state (`tri_and` fail-closed: no >
  uncertain > yes) derived from enumerated atoms.

### Priority-1 cleanup — commentary `is_social_norm` refactor

`scripts/commentary_occurred_event_contract_v3.py` gained
`validate_result_atomic`: the composite `is_social_norm` is now derived from
the four shared atomic scope fields (`weak_label_contract.derive_is_social_norm`)
and a contradictory supplied value is rejected. The legacy `validate_result`
entry point is unchanged so existing ledgers still validate.

### Milestone B — standardized LFs (`scripts/labeling_functions_v1.py`)

- The section-12.3 record interface (`make_lf_record` / `validate_lf_record`):
  stable `lf_id`, one explicit target per vote, `+1/-1/0` votes, mandatory
  `abstain_reason` on abstention, evidence-family grouping (section 12.5
  families), immutable provenance fields, optional artifact SHA-256.
- `registry_lf_records` wraps `config/audited_weak_signals_v1.json`:
  - **failed-transfer rules abstain by construction**
    (`failed_transfer_registry_enforced`);
  - retrieval/ranking-only rules abstain
    (`retrieval_or_ranking_use_only_never_a_label`) — query/title is never a
    label;
  - only the three audited exclusion/reroute rules
    (`witnessed_authority_exact_span_v3`, `witnessed_creator_staging_title_v2`,
    `witnessed_official_wwyd_channel_v1`) cast a real vote, and only `-1` on
    the `independent_bystander_signal` subtype target.
- `atomic_contract_lf_records` converts the
  `cross_pillar_shadow_supervision_v1` family votes into one LF per evidence
  family (correlated paraphrases cannot multiply).
- `eligibility_gate` implements the section-12.6 deterministic gates
  (media decodes, ordered bounds, demonstration/event interval present,
  sanitization verified when required); unknown gates fail closed.
- `write_lf_records` is an append-only shard writer that refuses overwrite.

### Milestone B pass condition — diagnostics (`scripts/lf_matrix_v1.py`)

Builds per-(pillar, target) matrices from LF record JSONL and reports per-LF
coverage/polarity, pairwise overlap/conflict/correlation, flags correlated
same-family pairs, and — given a manual gold ledger — per-LF precision/recall
with Wilson lower bounds. Duplicate conflicting votes for one (item, LF) cell
are an input error. CLI:

```bash
python scripts/lf_matrix_v1.py --records lf_records.jsonl \
    [--gold gold.jsonl] --out diagnostics.json
```

### Milestone C — shadow label model (`scripts/label_model_v1.py`)

- Family collapse first (any `-1` in a family wins, else `+1`, else abstain),
  so the generative model sees at most one vote per evidence family.
- `FamilyLabelModel`: EM-fit naive-Bayes-with-abstention over family votes
  (per-family accuracy + class prior, abstains treated as missing evidence,
  accuracies clamped to [0.05, 0.95]). Pure stdlib, deterministic, no RNG.
- `run_experiment` follows section 12.7: deterministic hash-based
  source-disjoint split (`source_disjoint_split`, grouped by source/cluster
  key), fit on train only, reliability-bin calibration + ECE on the
  calibration split, evaluation on the untouched test split against
  **majority vote** and **strongest single LF chosen on train only**.
- `shadow_band` implements the section-12.8 output policy:
  `high_confidence_candidate` / `disagreement_manual_review` /
  `likely_failure_or_reroute` / `insufficient_evidence_abstain`, plus
  `gate_failed_review`. **Gates dominate the posterior** — twenty transcript
  votes cannot compensate for a missing demonstration interval.
- Every output row carries `acceptance_label: null`,
  `corpus_disposition: null`, `delete_media: false`, `shadow_only: true`.
  CLI (`--out` posterior JSONL + `--summary-out`) refuses existing outputs.

### Tests

46 new tests in `tests/test_target_ontology_v1.py`,
`tests/test_labeling_functions_v1.py`, `tests/test_lf_matrix_v1.py`,
`tests/test_label_model_v1.py`, `tests/test_commentary_atomic_scope_adapter.py`
(vote/abstention validation, registry abstention enforcement, gate dominance,
EM recovery on synthetic corpora, split disjointness, calibration math,
append-only writers, legacy compatibility). Full local suite at this
checkpoint: 1,522 passed; 4 pre-existing environment failures in
`test_package_fresh_three_pillar_review_bundle.py` caused by the local ffmpeg
lacking a `libopus` encoder (unrelated to these changes; passes where the
canonical `envs/norm-scraper` ffmpeg is used).

---

## Update 2026-08-17 (second pass): shadow votes and gold tooling

### Governed `shadow_lf_vote` registry use

`scripts/weak_signal_registry.py` now validates a sixth declared use,
`shadow_lf_vote`, with a mandatory per-rule contract (target, signed vote,
evidence family, audited precision/recall, `shadow_only: true`). Five
mechanisms were promoted into it on their existing frozen audits (see
`docs/audited_weak_signal_registry_v1.md`): the intervention scan (+1 on
`reaction_grounded`), non-explanation polarity (+1 on `demonstration_present`,
precision recorded null because it was unstable across cohorts), and the three
exclusion cues (−1 on `independent_bystander_signal` only).
`labeling_functions_v1.registry_lf_records` is now registry-driven — the
hardcoded vote map was removed, and a rule votes only when the registry
carries a valid contract and the rule triggers. Failed-transfer rules and
retrieval/ranking-only rules abstain exactly as before.

### Gold-cohort wave tooling (Priority 4)

`scripts/build_gold_cohort_manifest_v1.py` builds a frozen audit wave per
section 13.4: deterministic seeded SHA-256 selection over an inventory JSONL;
strata `uniform` / `high_score` / `low_score` / `disagreement` /
`fresh_query` / `repair_candidate`; global source-, duplicate-cluster-, and
channel-cap disjointness; prior-UID exclusion; shortfalls reported, never
padded. Outputs (append-only, hash-sealed): `manifest.jsonl`,
`blind_ledger.jsonl` (atomic visual fields only — no transcript, label,
query, title, or score can leak into the blind pass), `reveal_ledger.jsonl`
(semantic fields, gated on `blind_row_frozen`), and `summary.json` with the
manifest SHA-256. The reviewer fills only atoms; every revised target is
derived in code by `target_ontology_v1`.

To launch the first wave: export an inventory JSONL from sk3 (uid, pillar,
source_uid, channel, cluster, strata tags, media_ref), run the builder with a
recorded seed, and freeze the manifest before rendering any review media.

## Update 2026-08-20 (third pass): commentary unlock and band hardening

- Commentary text LFs v2 scan a ±20 s transcript window around each statement
  (the stored stance quote rarely contains the event description); hypothetical
  framing is still judged on the quote alone.
- Registry promotions: `commentary_capture_title_event_v1` (+1
  `event_present_in_source`, audited precision 0.60) and
  `commentary_dual_vlm_retrieval_core_v1` (+1, precision 1.0 / recall 0.39;
  abstains wherever no dual-VLM score exists — none corpus-wide yet). Seven
  rules now hold `shadow_lf_vote`.
- Feature repair: 23,430 commentary low-level rows whose July backfill failed
  with `FileNotFoundError` (media landed later via the retention backfill)
  were re-scored on sk3 CPU shards; 797 true decode failures remain. The
  runner picks up `commentary_low_level_rerun*.jsonl` shards automatically.
- Band hardening: `high_confidence_candidate` additionally requires at least
  one positive family vote. The clamped 0.99 fitted prior had let zero-support
  and negative-only items into high bands (v3 witnessed bystander, v4
  commentary); prior-carried items now abstain. Regression-tested.
- Final corpus fit is `data/shadow_scores/20260820_lf_snorkel_v5/` (891k LF
  records over 51,837 sources). Commentary occurred-event videos 878 → 10,386;
  event-present videos 847 → 1,661; text+visual intersection 42 → 943.
  Fitted priors remain uncalibrated pending the gold wave.

## What this unlocks (Milestone B/C pass conditions)

- No failed-transfer rule can emit a live vote (enforced in code, tested).
- No correlated family is counted twice (family collapse before fitting;
  same-family correlation is flagged in diagnostics).
- All votes carry reproducible provenance and versions.
- The label model must beat majority vote and the strongest single LF on an
  untouched source-disjoint test split before any promotion discussion.

## What remains (not implementable locally)

1. **Materialize real LF records** on sk3 from the existing shadow-score
   ledgers (`data/shadow_scores/…`) and atomic contract outputs — one batched
   run, append-only shards under `data/shadow_scores/`.
2. **Larger source-disjoint human gold cohorts** (roadmap Priority 4),
   especially genuine witnessed reaction positives across responder roles —
   requires manual audit time, not code.
3. **Fit and calibrate per pillar/target on real data** with the section-12.7
   procedure; audit all predicted positives plus sampled negatives before any
   Milestone-E promotion.
4. **Corpus-scale posterior backfill** (Milestone D) in resumable shards on
   sk3, honoring GPU blocklists and parsimonious-SSH rules.
5. Multi-head critic, pairwise ranking, PU learning, cluster-aware splits
   (roadmap section 15) — deferred until the label model has audited gold.
6. Ops items: revive null-verified scheduling fairness checks, run/audit the
   18 ordinary-norm canaries, close the Appendix A.12 inventory gaps with one
   batched read-only check per host.
