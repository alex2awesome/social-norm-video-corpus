#!/usr/bin/env bash
# Resume the already-rendered instructional/witnessed audits and corrected
# commentary run under one managed VLM lease.

set -uo pipefail

bundle=$1
stage=$2
run=$3
audit_python=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/python
endpoint=http://127.0.0.1:8271/v1
model=qwen3-vl-8b-instruct
status=0

"$audit_python" "$bundle/scripts/run_instructional_temporal_critic_v4.py" \
  --manifest "$run/instructional/storyboard_manifest.jsonl" \
  --semantic-manifest "$stage/outputs/instructional_v4/sealed_selection.jsonl" \
  --out "$run/instructional/qwen_temporal_critic_v4.jsonl" \
  --endpoint "$endpoint" --model "$model" --workers 2 --timeout 300 --retries 2 \
  || status=1

"$audit_python" "$bundle/scripts/run_witnessed_reaction_candidate_av_vlm.py" \
  --manifest "$run/witnessed/v6_model_manifest.jsonl" \
  --out "$run/witnessed/qwen_role_causal_binding_v6.jsonl" \
  --endpoint "$endpoint" --model "$model" --workers 2 --timeout 300 --retries 2 \
  --media video --prompt-version v6_role_causal_binding \
  || status=1

if [[ -f "$run/commentary_v2/rendered/sealed_render_manifest.jsonl" ]]; then
  "$audit_python" "$bundle/scripts/run_commentary_temporal_verifier_vlm.py" \
    --manifest "$run/commentary_v2/rendered/sealed_render_manifest.jsonl" \
    --out "$run/commentary_v2/qwen_temporal_core_event_v3.jsonl" \
    --endpoint "$endpoint" --model "$model" --workers 2 --timeout 300 --retries 2 \
    --policy core_event_v3 \
    || status=1
fi

exit "$status"
