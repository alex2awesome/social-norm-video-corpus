#!/usr/bin/env bash
# Resume from already-rendered shadow artifacts after a fixed-GPU wait. This
# script never edits corpus files and uses only the explicit sk3 GPU allow-list.

set -uo pipefail

repo_root=$1
stage=$2
bundle=$3
run="$stage/fresh_run_v1"
audit_python=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/python
visual_python=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
ffmpeg=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/ffmpeg
ffprobe=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/ffprobe
commentary="$run/commentary_v2"
status=0

export PYTHONPATH="$bundle:$repo_root"
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME=/lfs/skampere3/0/alexspan/.cache/torch

if [[ ! -s "$run/instructional/storyboard_manifest.jsonl" ]] || \
   [[ ! -s "$run/witnessed/v6_model_manifest.jsonl" ]]; then
  echo "required rendered resume inputs are missing" >&2
  exit 2
fi
if [[ -e "$commentary" ]]; then
  echo "refusing to overwrite corrected commentary run: $commentary" >&2
  exit 2
fi
mkdir -p "$commentary"

labels="$bundle/inputs/commentary_accepted_visual_search_labels.jsonl"
if "$audit_python" "$bundle/scripts/index_commentary_selected_sources_v2.py" \
  --labels "$labels" --video-dir "$repo_root/data/discussion_video" \
  --ffprobe "$ffprobe" --out "$commentary/video_index.jsonl" \
  --failures "$commentary/video_index_failures.jsonl" \
  --summary "$commentary/video_index_summary.json" \
&& "$audit_python" "$bundle/scripts/build_commentary_hierarchical_source_packets_v2.py" \
  --labels "$labels" --metadata-dir "$repo_root/data/discussion" \
  --transcript-dir "$repo_root/data/transcripts" \
  --video-index "$commentary/video_index.jsonl" \
  --out "$commentary/source_packets.jsonl" \
&& "$audit_python" "$bundle/scripts/export_commentary_hierarchical_all_windows_v2.py" \
  --sources "$commentary/source_packets.jsonl" \
  --out "$commentary/all_windows.jsonl" \
  --summary "$commentary/all_windows_summary.json"; then
  audit_gpu=$("$audit_python" - <<'PY'
import subprocess
allowed = [0, 5, 6, 7]
free = {}
for gpu in allowed:
    value = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()[0]
    free[gpu] = int(value)
print(max(allowed, key=lambda gpu: (free[gpu], -allowed.index(gpu))))
PY
  )
  export CUDA_VISIBLE_DEVICES="$audit_gpu"
  if "$visual_python" "$bundle/scripts/score_visual_scene_baselines.py" \
    --manifest "$commentary/all_windows.jsonl" \
    --out "$commentary/baseline_scores.jsonl" \
    --frames 12 --device cuda --keypoints --clip --xclip --torch-threads 4 \
    --torch-home "$TORCH_HOME" --hf-home "$HF_HOME" \
  && "$audit_python" "$bundle/scripts/adapt_commentary_hierarchical_feature_scores_v2.py" \
    --baseline-scores "$commentary/baseline_scores.jsonl" \
    --out "$commentary/ranking_scores.jsonl" \
  && "$audit_python" "$bundle/scripts/build_commentary_hierarchical_window_manifest_v2.py" \
    --sources "$commentary/source_packets.jsonl" \
    --scores "$commentary/ranking_scores.jsonl" \
    --all-windows-out "$commentary/all_windows_verified.jsonl" \
    --selected-out "$commentary/selected_windows.jsonl" \
    --summary-out "$commentary/window_selection_summary.json" \
    --window-sec 12 --stride-sec 8 --per-route 2 --max-per-source 14 \
  && "$visual_python" "$bundle/scripts/render_commentary_hierarchical_windows_v2.py" \
    --selected "$commentary/selected_windows.jsonl" \
    --out-dir "$commentary/rendered" --ffmpeg "$ffmpeg" \
    --fps 3 --max-frames-per-window 36 --cell-width 256 --cell-max-height 256; then
    echo "CORRECTED_COMMENTARY_VISUAL_CANDIDATES_READY"
  else
    echo "CORRECTED_COMMENTARY_VISUAL_PIPELINE_FAILED" >&2
    status=1
  fi
else
  echo "CORRECTED_COMMENTARY_SOURCE_PIPELINE_FAILED" >&2
  status=1
fi

unset CUDA_VISIBLE_DEVICES
"$audit_python" "$bundle/scripts/run_managed_vllm_batch.py" \
  --profiles "$bundle/config/sk3_audit_vllm_profiles.json" \
  --profile qwen3_vl_8b_low_impact \
  --startup-timeout 1200 --gpu-wait-timeout 21600 --shutdown-timeout 120 \
  --batch-manifest "$run/witnessed/v6_model_manifest.jsonl" \
  --batch-output "$run/witnessed/qwen_role_causal_binding_v6.jsonl" \
  --min-batch-items 1 --max-batch-delay-seconds 0 -- \
  /bin/bash "$bundle/scripts/run_fresh_three_pillar_vlm_clients_resume_20260806.sh" \
    "$bundle" "$stage" "$run" \
  || status=1

"$audit_python" - "$run" "$status" <<'PY'
import json
import sys
from pathlib import Path
run = Path(sys.argv[1])
def count(path):
    return sum(bool(line.strip()) for line in path.open()) if path.is_file() else 0
summary = {
    "kind": "fresh_three_pillar_intensive_audit_batch_resume_20260806",
    "exit_status": int(sys.argv[2]),
    "instructional_storyboards": count(run / "instructional/storyboard_manifest.jsonl"),
    "instructional_vlm_outputs": count(run / "instructional/qwen_temporal_critic_v4.jsonl"),
    "witnessed_manual_media": count(run / "witnessed/manual_media_manifest.jsonl"),
    "witnessed_vlm_outputs": count(run / "witnessed/qwen_role_causal_binding_v6.jsonl"),
    "commentary_indexed_sources": count(run / "commentary_v2/video_index.jsonl"),
    "commentary_selected_windows": count(run / "commentary_v2/selected_windows.jsonl"),
    "commentary_vlm_outputs": count(run / "commentary_v2/qwen_temporal_core_event_v3.jsonl"),
    "dynamic_gpu_allow_list": [0, 5, 6, 7],
    "automatic_acceptance": False,
    "corpus_mutation_authorized": False,
}
(run / "resume_batch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

exit "$status"
