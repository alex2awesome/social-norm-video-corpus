#!/usr/bin/env bash
set -euo pipefail

audit_root=/lfs/skampere3/0/alexspan/norm-scraper
audit_python=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/python
audit_ffmpeg=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/ffmpeg
audit_ffprobe=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/ffprobe
audit_dir="$audit_root/audit_runs/20260811_strict_audit_calibration_v1"

cd "$audit_root"

"$audit_python" scripts/render_corpus_shadow_proxies.py \
  --manifest "$audit_dir/model_manifest.jsonl" \
  --out-dir "$audit_dir/model_proxies" \
  --records "$audit_dir/model_proxy_records.jsonl" \
  --complete-manifest "$audit_dir/model_manifest_with_proxies.jsonl" \
  --ffmpeg "$audit_ffmpeg" \
  --ffprobe "$audit_ffprobe" \
  --workers 4 \
  --max-fps 2.0 \
  --max-frames 96 \
  --max-side 448

"$audit_python" scripts/run_managed_vllm_batch.py \
  --profiles config/sk3_audit_vllm_profiles.json \
  --profile qwen3_vl_8b_low_impact \
  --startup-timeout 900 \
  --gpu-wait-timeout 1800 \
  --shutdown-timeout 120 \
  --batch-manifest "$audit_dir/model_manifest_with_proxies.jsonl" \
  --batch-output "$audit_dir/vlm/qwen_pillar_complete.jsonl" \
  --min-batch-items 1 \
  --max-batch-delay-seconds 0 \
  -- \
  "$audit_python" scripts/run_strict_calibration_vlm_batch_v1.py \
    --python "$audit_python" \
    --runner "$audit_root/scripts/run_open_vlm_scene_benchmark.py" \
    --manifest "$audit_dir/model_manifest_with_proxies.jsonl" \
    --out-dir "$audit_dir/vlm" \
    --completion-output "$audit_dir/vlm/qwen_pillar_complete.jsonl" \
    --workers 2
