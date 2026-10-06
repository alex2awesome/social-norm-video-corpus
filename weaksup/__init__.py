"""Weak-supervision layer for the social-norms corpus (2026-08/09).

Pipeline order:
  1. target_ontology_v1        - revised targets, deterministic tri-state derivations
  2. weak_label_contract       - shared atomic social-scope schema (legacy, canonical)
  3. weak_signal_registry      - audited-mechanism governance incl. shadow_lf_vote
  4. labeling_functions_v1     - standardized LF records, gates, registry votes
  5. materialize_corpus_lf_records_v1 / visual_feature_lfs_v1 - corpus -> LF records
  6. lf_matrix_v1              - matrices + coverage/conflict/correlation diagnostics
  7. label_model_v1            - EM family label model, bands, splits, calibration
  8. run_corpus_lf_snorkel_v1  - corpus-wide orchestration (sk3)
  9. witnessed_action_window_proposer_v1 + cut_witnessed_action_clips_v1
                               - pre-reaction violation windows -> action/context clips
 10. export_instructional_demo_tier_v1 / propose_commentary_event_windows_v1
                               - per-pillar dataset views
 11. build_gold_cohort_manifest_v1 / reaction_language_tags_v1 - audit tooling

scripts/ keeps thin re-export shims for every module, so legacy imports
(`from scripts.label_model_v1 import ...`) and sk3 command lines are unchanged.
Audited legacy scorers (score_witnessed_*, witnessed_reaction_*) deliberately
stay in scripts/ untouched.
"""
