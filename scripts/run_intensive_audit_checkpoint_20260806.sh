#!/usr/bin/env bash
# One-shot, read-mostly checkpoint for the 2026-08-06 intensive audits.
#
# The only writes are derived shadow-audit artifacts. Source clips, metadata,
# corpus dispositions, and query state are never changed. Existing audit
# directories are never overwritten.

set -euo pipefail

repo_root=/lfs/skampere3/0/alexspan/norm-scraper
audit_python=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/python
visual_python=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
audit_ffmpeg=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/ffmpeg
audit_ffprobe=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/ffprobe

witness_root="$repo_root/audit_runs/20260806_witnessed_video_asr_corpus_intensive_v1"
witness_manifest="$repo_root/data/shadow_scores/20260806_witnessed_video_asr_v3/manifest_retry.jsonl"
witness_scores="$repo_root/data/shadow_scores/20260806_witnessed_video_asr_v3/qwen_video_asr_v3.jsonl"
witness_pid_file="$witness_root/model_score_retry_pid.json"
witness_selection="$witness_root/selection_v1"

cd "$repo_root"

witness_alive=no
if [[ -f "$witness_pid_file" ]]; then
  witness_pid=$(
    "$audit_python" -c \
      'import json,sys; print(int(json.load(open(sys.argv[1]))["pid"]))' \
      "$witness_pid_file"
  )
  if [[ -r "/proc/$witness_pid/cmdline" ]] && \
      tr '\0' ' ' < "/proc/$witness_pid/cmdline" | \
      grep -q 'run_witnessed_reaction_candidate_av_vlm.py'; then
    witness_alive=yes
  fi
fi
echo "WITNESS_SCORER_ALIVE=$witness_alive"

witness_summary=$(
  "$audit_python" scripts/summarize_witnessed_score_failures.py \
    --manifest "$witness_manifest" \
    --scores "$witness_scores" \
    --model qwen3-vl-8b-instruct
)
echo "$witness_summary"

if [[ "${ENABLE_SK3_QWEN_RETRY:-0}" == 1 ]] && \
    [[ "$witness_alive" == no ]] && ! \
    "$audit_python" -c \
      'import json,sys; value=json.loads(sys.argv[1]); raise SystemExit(0 if value["successful_coverage_fraction"] >= .98 else 1)' \
      "$witness_summary"; then
  echo "WITNESS_RETRY_START=outstanding_successful_ids_only"
  set +e
  "$audit_python" scripts/run_managed_vllm_batch.py \
    --profiles config/sk3_audit_vllm_profiles.json \
    --profile qwen3_vl_8b_low_impact \
    --startup-timeout 900 \
    --gpu-wait-timeout 1800 \
    --shutdown-timeout 120 \
    --batch-manifest "$witness_manifest" \
    --batch-output "$witness_scores" \
    --min-batch-items 1 \
    --max-batch-delay-seconds 0 -- \
    "$audit_python" scripts/run_witnessed_reaction_candidate_av_vlm.py \
      --manifest "$witness_manifest" \
      --out "$witness_scores" \
      --endpoint http://127.0.0.1:8271/v1 \
      --model qwen3-vl-8b-instruct \
      --workers 2 \
      --timeout 300 \
      --retries 2 \
      --media video \
      --prompt-version v3_audiovisual_wording
  witness_retry_code=$?
  set -e
  echo "WITNESS_RETRY_EXIT_CODE=$witness_retry_code"
  witness_summary=$(
    "$audit_python" scripts/summarize_witnessed_score_failures.py \
      --manifest "$witness_manifest" \
      --scores "$witness_scores" \
      --model qwen3-vl-8b-instruct
  )
  echo "$witness_summary"
fi

if [[ "$witness_alive" == no ]] && \
    "$audit_python" -c \
      'import json,sys; value=json.loads(sys.argv[1]); raise SystemExit(0 if value["successful_coverage_fraction"] >= .90 and not value["unexpected_candidate_ids"] else 1)' \
      "$witness_summary"; then
  if "$audit_python" -c \
      'import json,sys; value=json.loads(sys.argv[1]); raise SystemExit(0 if value["successful_coverage_fraction"] >= .98 else 1)' \
      "$witness_summary"; then
    echo "WITNESS_POPULATION_COVERAGE_GATE=pass"
  else
    echo "WITNESS_POPULATION_COVERAGE_GATE=fail_audit_may_proceed_no_promotion"
  fi
  if [[ ! -e "$witness_selection" ]]; then
    "$audit_python" scripts/select_witnessed_video_asr_corpus_audit.py \
      --manifest "$witness_manifest" \
      --scores "$witness_scores" \
      --exclude-uids "$witness_root/excluded_prior_manual_uids.txt" \
      --uniform-clips 60 \
      --positive-enrichment 30 \
      --near-miss-enrichment 30 \
      --model-error-enrichment 10 \
      --out-dir "$witness_selection"
    set +e
    "$visual_python" scripts/materialize_witnessed_corpus_audit_media.py \
      --selection "$witness_selection/sealed_selection.jsonl" \
      --out-dir "$witness_selection/manual_media" \
      --manifest "$witness_selection/manual_media_manifest.jsonl" \
      --failures "$witness_selection/manual_media_failures.jsonl" \
      --summary "$witness_selection/manual_media_summary.json" \
      --ffmpeg "$audit_ffmpeg" \
      --ffprobe "$audit_ffprobe" \
      --storyboard-frames 24
    witness_render_code=$?
    set -e
    echo "WITNESS_AUDIT_RENDER_EXIT_CODE=$witness_render_code"
  fi
  echo "WITNESS_AUDIT_DIR=$witness_selection"
else
  echo "WITNESS_AUDIT_DEFERRED=score_coverage_or_process_gate"
fi

instructional_source=$(
  "$audit_python" - "$repo_root/data/shadow_scores/20260724_full_corpus_v1" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = "a54a62ec6f6f18feef820e8e5090faee97f21cd7961dc2dad59379bed44efc8b"
required = {"item_id", "uid", "source_clip", "duration_hint", "query_source"}
for path in sorted(root.glob("*.jsonl")):
    try:
        with path.open() as handle:
            first = next((line for line in handle if line.strip()), "")
        if not first or not required.issubset(json.loads(first)):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == expected:
            print(path)
            raise SystemExit(0)
    except (OSError, ValueError, json.JSONDecodeError):
        continue
raise SystemExit("frozen 42,525-row instructional source manifest not found")
PY
)
echo "INSTRUCTIONAL_SOURCE=$instructional_source"

instructional_out="$repo_root/audit_runs/20260806_instructional_retro_duration_holdout_v1"
if [[ ! -e "$instructional_out" ]]; then
  "$audit_python" scripts/select_instructional_retro_duration_holdout_v1.py \
    --source "$instructional_source" \
    --exclude-uids "$repo_root/audit_runs/20260728_instructional_v9_corpus_validation_v1/excluded_prior_manual_uids.txt" \
    --exclude-jsonl "$repo_root/audit_runs/20260805_full_signal_expansion_audit_v1/instructional_v20_sample_v2/sealed_selection.jsonl" \
    --exclude-jsonl "$repo_root/audit_runs/20260805_instructional_v22_fresh_holdout_v1/sealed_selection.jsonl" \
    --exclude-jsonl "$repo_root/audit_runs/20260806_instructional_title_v23_union_v1/fresh_holdout_100/sealed_selection.jsonl" \
    --exclude-jsonl "$repo_root/audit_runs/20260806_instructional_blind_episode_relaxed_transfer_v1/fresh_holdout_100/sealed_selection.jsonl" \
    --exclude-jsonl "$repo_root/audit_runs/20260806_instructional_demo_consensus_v2_fresh_transfer/fresh_holdout_100/sealed_selection.jsonl" \
    --exclude-jsonl "$repo_root/audit_runs/20260806_instructional_retro_duration_refinement_v1/normalized_records.jsonl" \
    --signal-positive 10 \
    --short-retro 6 \
    --nonretro-control 16 \
    --out "$instructional_out"
  "$visual_python" scripts/render_instructional_v9_corpus_storyboards.py \
    --manifest "$instructional_out/blind_source_manifest.jsonl" \
    --out-dir "$instructional_out/rendered" \
    --out-manifest "$instructional_out/storyboard_manifest.jsonl" \
    --frames 36 \
    --workers 8
fi
echo "INSTRUCTIONAL_AUDIT_DIR=$instructional_out"
