#!/usr/bin/env bash
# One detached, read-only-to-corpus audit batch.  CPU/GPU models load only for
# this bounded batch and the managed vLLM process is always unloaded afterward.

set -uo pipefail

repo_root=$1
stage=$2
bundle=$3
run="$stage/fresh_run_v1"
audit_python=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/python
visual_python=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
ffmpeg=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/ffmpeg
ffprobe=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/ffprobe
status=0

export PYTHONPATH="$bundle:$repo_root"
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME=/lfs/skampere3/0/alexspan/.cache/torch
export CUDA_VISIBLE_DEVICES=7

if [[ -e "$run" ]]; then
  echo "refusing to overwrite existing audit run: $run" >&2
  exit 2
fi
mkdir -p "$run/instructional" "$run/witnessed" "$run/commentary"

# Instructional: render every sealed fresh clip as a metadata-blind 36-frame
# storyboard. No detector label is present in the pixels passed to Stage A.
if ! "$visual_python" "$bundle/scripts/render_instructional_v9_corpus_storyboards.py" \
  --manifest "$stage/outputs/instructional_v4/blind_source_manifest.jsonl" \
  --out-dir "$run/instructional/rendered" \
  --out-manifest "$run/instructional/storyboard_manifest.jsonl" \
  --frames 36 --workers 8; then
  echo "INSTRUCTIONAL_RENDER_FAILED" >&2
  status=1
fi

# Witnessed: recut every selected candidate from its original clip, preserving
# source audio and producing a blinded 24-frame storyboard for human review.
if "$visual_python" "$bundle/scripts/materialize_witnessed_corpus_audit_media.py" \
  --selection "$stage/outputs/witnessed_v6/sealed_selection.jsonl" \
  --out-dir "$run/witnessed/manual_media" \
  --manifest "$run/witnessed/manual_media_manifest.jsonl" \
  --failures "$run/witnessed/manual_media_failures.jsonl" \
  --summary "$run/witnessed/manual_media_summary.json" \
  --ffmpeg "$ffmpeg" --ffprobe "$ffprobe" --storyboard-frames 24; then
  "$audit_python" "$bundle/scripts/adapt_witnessed_manual_media_v6.py" \
    --selection "$stage/outputs/witnessed_v6/sealed_selection.jsonl" \
    --media-manifest "$run/witnessed/manual_media_manifest.jsonl" \
    --out "$run/witnessed/v6_model_manifest.jsonl" || status=1
else
  echo "WITNESSED_MEDIA_FAILED" >&2
  status=1
fi

# Commentary: only the fully manually text-reviewed labels enter full-source
# tiling. Cheap models rank windows but cannot accept or reject a video.
commentary_labels="$bundle/inputs/commentary_accepted_visual_search_labels.jsonl"
if "$audit_python" "$bundle/scripts/index_commentary_selected_sources_v2.py" \
  --labels "$commentary_labels" --video-dir "$repo_root/data/discussion_video" \
  --ffprobe "$ffprobe" \
  --out "$run/commentary/video_index.jsonl" \
  --failures "$run/commentary/video_index_failures.jsonl" \
  --summary "$run/commentary/video_index_summary.json"; then
  if [[ -s "$run/commentary/video_index.jsonl" ]]; then
    if "$audit_python" "$bundle/scripts/build_commentary_hierarchical_source_packets_v2.py" \
      --labels "$commentary_labels" \
      --metadata-dir "$repo_root/data/discussion" \
      --transcript-dir "$repo_root/data/transcripts" \
      --video-index "$run/commentary/video_index.jsonl" \
      --out "$run/commentary/source_packets.jsonl" \
    && "$audit_python" "$bundle/scripts/export_commentary_hierarchical_all_windows_v2.py" \
      --sources "$run/commentary/source_packets.jsonl" \
      --out "$run/commentary/all_windows.jsonl" \
      --summary "$run/commentary/all_windows_summary.json" \
    && "$visual_python" "$bundle/scripts/score_visual_scene_baselines.py" \
      --manifest "$run/commentary/all_windows.jsonl" \
      --out "$run/commentary/baseline_scores.jsonl" \
      --frames 12 --device cuda --keypoints --clip --xclip --torch-threads 4 \
      --torch-home "$TORCH_HOME" --hf-home "$HF_HOME" \
    && "$audit_python" "$bundle/scripts/adapt_commentary_hierarchical_feature_scores_v2.py" \
      --baseline-scores "$run/commentary/baseline_scores.jsonl" \
      --out "$run/commentary/ranking_scores.jsonl" \
    && "$audit_python" "$bundle/scripts/build_commentary_hierarchical_window_manifest_v2.py" \
      --sources "$run/commentary/source_packets.jsonl" \
      --scores "$run/commentary/ranking_scores.jsonl" \
      --all-windows-out "$run/commentary/all_windows_verified.jsonl" \
      --selected-out "$run/commentary/selected_windows.jsonl" \
      --summary-out "$run/commentary/window_selection_summary.json" \
      --window-sec 12 --stride-sec 8 --per-route 2 --max-per-source 14 \
    && "$visual_python" "$bundle/scripts/render_commentary_hierarchical_windows_v2.py" \
      --selected "$run/commentary/selected_windows.jsonl" \
      --out-dir "$run/commentary/rendered" --ffmpeg "$ffmpeg" \
      --fps 3 --max-frames-per-window 36 --cell-width 256 --cell-max-height 256; then
      echo "COMMENTARY_VISUAL_CANDIDATES_READY"
    else
      echo "COMMENTARY_VISUAL_PIPELINE_FAILED" >&2
      status=1
    fi
  fi
else
  echo "COMMENTARY_SOURCE_INDEX_FAILED" >&2
  status=1
fi

# One managed Qwen lease serves all three bounded cohorts, then is unloaded.
if [[ -s "$run/witnessed/v6_model_manifest.jsonl" ]]; then
  "$audit_python" "$bundle/scripts/run_managed_vllm_batch.py" \
    --profiles "$bundle/config/sk3_audit_vllm_profiles.json" \
    --profile qwen3_vl_8b_low_impact \
    --startup-timeout 1200 --gpu-wait-timeout 21600 --shutdown-timeout 120 \
    --batch-manifest "$run/witnessed/v6_model_manifest.jsonl" \
    --batch-output "$run/witnessed/qwen_role_causal_binding_v6.jsonl" \
    --min-batch-items 1 --max-batch-delay-seconds 0 -- \
    /bin/bash "$bundle/scripts/run_fresh_three_pillar_vlm_clients_20260806.sh" \
      "$bundle" "$stage" "$run" "$repo_root" \
    || status=1
else
  echo "MANAGED_VLM_SKIPPED_NO_WITNESSED_MANIFEST" >&2
  status=1
fi

"$audit_python" - "$run" "$status" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
status = int(sys.argv[2])
def count(path):
    return sum(bool(line.strip()) for line in path.open()) if path.is_file() else 0
summary = {
    "kind": "fresh_three_pillar_intensive_audit_batch_20260806",
    "exit_status": status,
    "instructional_storyboards": count(run / "instructional/storyboard_manifest.jsonl"),
    "instructional_vlm_outputs": count(run / "instructional/qwen_temporal_critic_v4.jsonl"),
    "witnessed_manual_media": count(run / "witnessed/manual_media_manifest.jsonl"),
    "witnessed_vlm_outputs": count(run / "witnessed/qwen_role_causal_binding_v6.jsonl"),
    "commentary_text_labels_entered": count(run / "commentary/video_index.jsonl"),
    "commentary_selected_windows": count(run / "commentary/selected_windows.jsonl"),
    "commentary_vlm_outputs": count(run / "commentary/qwen_temporal_core_event_v3.jsonl"),
    "automatic_acceptance": False,
    "corpus_mutation_authorized": False,
}
(run / "batch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

exit "$status"
