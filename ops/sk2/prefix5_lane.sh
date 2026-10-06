#!/bin/bash
# 5-pass prefix lane: wait for a genuinely free GPU, load Qwen only for this
# batch, and release the complete process group on every exit path.
set -uo pipefail

export HOME=/lfs/skampere2/0/alexspan
export HF_HOME=$HOME/.cache/huggingface
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -n 65536 2>/dev/null
D=$HOME/norm-research/datasets/prompt-optimality-test
VLLM=$HOME/miniconda3/bin/vllm; CACHE=/lfs/skampere2/0/shared_hf_cache/hub
cd $D || exit 9
L=logs/prefix_5pass_20260728.log

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
PORT=8191; M=Qwen3-8B
SNAP=$(ls -d $CACHE/models--Qwen--$M/snapshots/*/ | head -1)
echo "$(date -u +%FT%TZ) free gpu$GPU; loading $M port=$PORT memory_fraction=$GPU_MEMORY_UTILIZATION" |
  tee -a "$L"
setsid "$VLLM" serve "$SNAP" --served-model-name "$M" --port "$PORT" --host 127.0.0.1 \
  --max-model-len 32768 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --reasoning-parser qwen3 \
  --disable-log-requests > logs/vllm_prefix5.log 2>&1 &
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
  tail -20 logs/vllm_prefix5.log | tee -a "$L"
  exit 1
fi

echo "$(date -u +%FT%TZ) server_ready pid=$VP; starting batch" | tee -a "$L"
export DSPY_CACHEDIR=$HOME/dspy_cache_prefix5
./.venv/bin/python - <<PYRUN 2>&1 | tee -a $L
import dspy, sys
sys.argv = ["x", "http://127.0.0.1:$PORT/v1"]
lm = dspy.LM("openai/Qwen3-8B", api_base="http://127.0.0.1:$PORT/v1", api_key="EMPTY", cache=False,
             temperature=0.6, top_p=0.95, max_tokens=24000, num_retries=10, timeout=300,
             extra_body={"top_k": 20})
dspy.configure(lm=lm)
import paperexact_arms as px
px.EVAL_THREADS = 32
exec(open("prefix_5pass.py").read())
PYRUN
rc=${PIPESTATUS[0]}
echo "$(date -u +%FT%TZ) PREFIX 5PASS COMPLETE rc=$rc" | tee -a "$L"
exit "$rc"
