#!/bin/bash
# Budget-matched GEPA lane (2026-07-28): answers the reviewer question the paper currently
# discloses as open -- is the M_omega margin a BUDGET artifact? Runs official GEPA at 600 AND
# 2400 metric calls in ONE session, same server, same local reflection LM, k=5 final tests.
# Only the budget varies between the two arms.
set -uo pipefail

export HOME=/lfs/skampere2/0/alexspan
export HF_HOME=$HOME/.cache/huggingface
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -n 65536 2>/dev/null
D=$HOME/norm-research/datasets/prompt-optimality-test
VLLM=$HOME/miniconda3/bin/vllm; CACHE=/lfs/skampere2/0/shared_hf_cache/hub
cd $D || exit 9
L=logs/budgetmatch_20260728.log

GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-300}
MAX_GPU_WAIT_MINUTES=${MAX_GPU_WAIT_MINUTES:-1440}
MODEL_START_TIMEOUT_SECONDS=${MODEL_START_TIMEOUT_SECONDS:-300}
SERIALIZE_QWEN_LANES=${SERIALIZE_QWEN_LANES:-1}
MODEL_LANE_LOCK=${MODEL_LANE_LOCK:-$HOME/.qwen3_8b_batch.lock}
VP=

cleanup() {
  rc=$?
  trap - EXIT
  if [ -n "${VP:-}" ] && kill -0 -- "-$VP" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) unloading vLLM pid=$VP" | tee -a "$L"
    kill -TERM -- "-$VP" 2>/dev/null || kill -TERM "$VP" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-$VP" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$VP" 2>/dev/null || true
    wait "$VP" 2>/dev/null || true
  fi
  echo "$(date -u +%FT%TZ) lane_exit rc=$rc gpu_released" | tee -a "$L"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if [ "$SERIALIZE_QWEN_LANES" = 1 ]; then
  exec 9>"$MODEL_LANE_LOCK"
  echo "$(date -u +%FT%TZ) waiting for shared Qwen3-8B batch lane" | tee -a "$L"
  flock 9
  echo "$(date -u +%FT%TZ) acquired shared Qwen3-8B batch lane" | tee -a "$L"
fi

waited=0
while true; do
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F", " '$2<2000{print $1; exit}')
  [ -n "$GPU" ] && break
  if [ "$waited" -ge "$((MAX_GPU_WAIT_MINUTES * 60))" ]; then
    echo "$(date -u +%FT%TZ) no free GPU after ${MAX_GPU_WAIT_MINUTES}m; leaving batch queued" |
      tee -a "$L"
    exit 75
  fi
  echo "$(date -u +%FT%TZ) no free GPU; delaying batch (${waited}s waited)" | tee -a "$L"
  sleep "$GPU_POLL_SECONDS"
  waited=$((waited + GPU_POLL_SECONDS))
done

export CUDA_VISIBLE_DEVICES=$GPU
PORT=8192; M=Qwen3-8B
SNAP=$(ls -d $CACHE/models--Qwen--$M/snapshots/*/ | head -1)
echo "$(date -u +%FT%TZ) budget-match lane on gpu$GPU port=$PORT memory_fraction=$GPU_MEMORY_UTILIZATION" |
  tee -a "$L"
setsid "$VLLM" serve "$SNAP" --served-model-name "$M" --port "$PORT" --host 127.0.0.1 \
  --max-model-len 32768 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --reasoning-parser qwen3 \
  --disable-log-requests > logs/vllm_budgetmatch.log 2>&1 &
VP=$!
UP=0
for _ in $(seq 1 "$((MODEL_START_TIMEOUT_SECONDS / 5))"); do
  sleep 5
  /usr/bin/curl -fsS -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
    grep -q "$M" && { UP=1; break; }
  kill -0 "$VP" 2>/dev/null || break
done
if [ "$UP" != 1 ]; then
  echo "$(date -u +%FT%TZ) SERVER FAILED" | tee -a "$L"
  tail -20 logs/vllm_budgetmatch.log | tee -a "$L"
  exit 1
fi

echo "$(date -u +%FT%TZ) server_ready pid=$VP; starting batch" | tee -a "$L"
export DSPY_CACHEDIR=$HOME/dspy_cache_budgetmatch
REFL="local:$M@http://127.0.0.1:$PORT/v1"
for B in 600 2400; do
  echo "$(date -u +%FT%TZ) === GEPA official budget=$B ===" | tee -a $L
  ./.venv/bin/python paperexact_arms.py hotpot --arm official \
     --task-lm openai/$M --api-base http://127.0.0.1:$PORT/v1 --lm-cache-off \
     --budget-calls $B --reflection-model "$REFL" --run-tag budgetmatch$B \
     --eval-threads 32 --test-passes 5 --max-tokens 24000 2>&1 | tail -25 | tee -a $L
  rc=${PIPESTATUS[0]}
  echo "$(date -u +%FT%TZ) budget=$B rc=$rc" | tee -a "$L"
  [ "$rc" -eq 0 ] || exit "$rc"
done
for B in 600 2400; do
  f=runs_paperexact/hotpot/Qwen3-8B/official_budgetmatch$B/result.json
  if [ -f "$f" ]; then echo "ARTIFACT ok $f: $(python3 -c "import json;j=json.load(open('$f'));print(j.get('seed_test'),j.get('best_test'),j.get('budget_calls'))")" | tee -a $L
  else echo "ARTIFACT MISSING $f" | tee -a $L; fi
done
echo "$(date -u +%FT%TZ) BUDGETMATCH COMPLETE rc=0" | tee -a "$L"
