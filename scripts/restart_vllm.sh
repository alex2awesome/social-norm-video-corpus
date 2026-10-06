#!/bin/bash
# (Re)launch the llama70b vLLM on 127.0.0.1:8017 if it is not responding.
# Picks the GPU with the most free memory. Idempotent: no-op if already healthy.
set -u
ENDPOINT="http://127.0.0.1:8017/v1/chat/completions"
PROJ=/lfs/skampere3/0/alexspan/norm-scraper
SNAP=/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362
cd "$PROJ" || exit 1

probe() {
  curl -s -m 8 "$ENDPOINT" -H "Content-Type: application/json" \
    -d '{"model":"llama70b","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' \
    2>/dev/null | grep -q '"choices"'
}

if probe; then echo "vllm healthy"; exit 0; fi
echo "vllm not responding; relaunching $(date)"

# free GPU = most free memory
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' ')
echo "selected GPU $GPU"

# kill stale pid if any
[ -f /tmp/vllm.pid ] && kill -9 "$(cat /tmp/vllm.pid)" 2>/dev/null
sleep 2

HOME=/lfs/skampere3/0/alexspan CUDA_VISIBLE_DEVICES="$GPU" FLASHINFER_DISABLE_VERSION_CHECK=1 nohup \
  /lfs/skampere3/0/alexspan/miniconda3/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$SNAP" --served-model-name llama70b --port 8017 --host 127.0.0.1 \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  > "$PROJ/vllm_llama.log" 2>&1 &
echo $! > /tmp/vllm.pid
echo "relaunched vllm pid $(cat /tmp/vllm.pid) on GPU $GPU"
