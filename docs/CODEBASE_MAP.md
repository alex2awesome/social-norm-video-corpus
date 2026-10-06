# Codebase map (2026-09-01)

## Layout

| Directory | What lives there |
| --- | --- |
| `weaksup/` | **The weak-supervision layer** (2026-08/09): ontology → LFs → label model → dataset views. Read `weaksup/__init__.py` for the pipeline order. |
| `src/` | Live collection pipeline (crawl, transcribe, detect, clip, audio events, state DB). Deployed on sk3; batch entry is `batch_pipeline.sh`. |
| `scripts/` | Everything else, three kinds: (a) thin **shims** re-exporting `weaksup/` modules (kept so `from scripts.x import` and sk3 command lines never break); (b) **audited legacy scorers** that registry evidence depends on — frozen, do not move: `score_witnessed_staging_cues.py`, `score_witnessed_authority_reaction_cues.py`, `witnessed_reaction_text_features.py` (v1 frozen, v2 additions appended), `witnessed_reaction_candidate_scan.py`, `score_witnessed_reaction_candidate_windows.py`, `score_commentary_title_event_cues.py`, `score_visual_scene_baselines.py`, `combine_audited_shadow_scores.py`; (c) ~130 historical audit/experiment scripts (build_*/evaluate_*/compile_* per audit run — reference only). |
| `config/` | `audited_weak_signals_v1.json` (the governance registry — single source of truth for which mechanisms may vote), settings, query plans, phrase tiers. |
| `tests/` | Pytest; `pytest.ini` puts `.` and `scripts` on the path. 4 known env-only failures (local ffmpeg lacks libopus). |
| `docs/` | `project_goals_progress_and_remaining_work_20260817.md` (frozen roadmap), `weak_supervision_lf_snorkel_implementation_v1.md` (implementation log), `audited_weak_signal_registry_v1.md`, this map. |
| `audit_runs/` | Frozen audit evidence (manifests, ledgers, reports, hashes). Registry rules point here. |
| `spotcheck/` | Local review webpage + preview clips (regenerated per wave; not canonical data). |

## Key data locations (sk3, under `/lfs/skampere3/0/alexspan/norm-scraper/`)

| Path | Contents |
| --- | --- |
| `data/shadow_scores/20260820_lf_snorkel_v5/` | Canonical LF records + posteriors (six pillar/target models) |
| `data/shadow_scores/20260828_action_windows_v2_full/` | 21,809 action-window proposals (any-high) |
| `data/action_clips_v1/` | Cut clips: `clips/` (tight action view) + `clips_context/` (30s context view) + shard manifests |
| `data/shadow_scores/20260828_instructional_demo_tier_v1/` | Instructional 0.80-tier manifest |
| `data/shadow_scores/20260901_commentary_event_windows_v2/` | Retimed commentary windows |
| `data/shadow_scores/20260901_reaction_language_tags_v1.jsonl` | Per-item language tags |
| `data/shadow_scores/20260828_dataset_views_rollup.json` | Dataset-view roll-up |

## Deployment

Rsync `weaksup/` AND the `scripts/` shims to sk3; run via
`<env-python> scripts/<name>.py ...` (shims bootstrap `sys.path`) or
`python -m weaksup.<name>` from the repo root. Envs: `norm-scraper` for
pipeline/ffmpeg work, `ai_usage` for cv2/vLLM stages. Honor the parsimonious-SSH
rule: one batched session per operation, nohup + shard for long jobs.
