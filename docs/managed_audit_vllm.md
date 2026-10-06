# Managed audit VLM batches on sk2

The Qwen3-VL-8B, GLM-4.6V-Flash, and Gemma-3-27B-IT audit servers are
intentionally not permanent services. They are used in long, resumable audit
batches and should release their H200 after each batch.

Use `scripts/run_managed_vllm_batch.py` with the matching profile. For example:

```bash
/lfs/skampere2/0/alexspan/miniconda3/bin/python \
  scripts/run_managed_vllm_batch.py \
  --profiles config/sk2_audit_vllm_profiles.json \
  --profile qwen3_vl_8b \
  --batch-manifest MANIFEST.jsonl \
  --batch-output OUTPUT.jsonl \
  --min-batch-items 50 \
  --max-batch-delay-seconds 3600 -- \
  /lfs/skampere2/0/alexspan/miniconda3/bin/python \
  scripts/run_open_vlm_v9a_storyboards.py \
  --manifest MANIFEST.jsonl --out OUTPUT.jsonl \
  --endpoint http://127.0.0.1:8271/v1 \
  --model qwen3-vl-8b-instruct
```

For GLM, select `--profile glm_4_6v_flash` and port 8272. For Gemma, select
`--profile gemma_3_27b_it` and port 8273.

The lifecycle is:

1. accumulate pending records without touching a GPU;
2. wait for the profile lock (ready batches serialize);
3. wait for the configured GPU to have enough free memory;
4. start vLLM and verify the exact model through `/v1/models`;
5. run the complete client batch; and
6. terminate the vLLM process group and release the GPU, even if the client
   exits nonzero or the wrapper is interrupted.

The audit clients append and resume JSONL output, so delaying a batch or
rerunning after a failure does not discard completed records. A server found
already running outside the lease is never killed automatically; the wrapper
stops only the process group that it started.

On Linux, both the batch client and server also receive a parent-death
`SIGTERM` safeguard before `exec`. This is a last-resort cleanup path if the
wrapper itself is killed abruptly; normal completion and handled signals still
use the process-group shutdown above. This prevents an orphaned zero-utilization
model from remaining resident after a lost remote wrapper.

Every invocation also appends process-safe lifecycle records to
`managed_servers/managed_vllm_events.jsonl`. These distinguish time spent
waiting for a batch, model cold-start time, client inference time, GPU lease
time, records added, clean unloads, empty skips, and failures. Summarize the
last day with:

```bash
/lfs/skampere2/0/alexspan/miniconda3/bin/python \
  scripts/report_managed_vllm_usage.py \
  /lfs/skampere2/0/alexspan/visual_audit/managed_servers/managed_vllm_events.jsonl \
  --hours 24
```

## Batch policy

Do not invoke a managed server for each clip. With `--batch-manifest`, the
wrapper itself waits without touching a GPU until at least 50 pending rows are
ready or one hour has elapsed. `--batch-output` subtracts completed rows, so an
already-finished resumable batch exits without loading the model. The pending
count is checked again after acquiring the profile lock in case an earlier
lease completed the same output.

The lease may then wait for its configured GPU for up to 24 hours by default.
This deliberately trades at most one hour of audit latency for avoiding
repeated model cold starts and idle GPU residency. High-priority manual
holdouts can omit `--batch-manifest` to run immediately, but should still run
as one batch.

The sk2 GLM and Gemma profiles disable only FlashInfer's optional top-k/top-p
sampler. This avoids an unreliable JIT-link step on a cold process; their
visual attention still uses FlashAttention. Qwen and GLM have been
cold-started through this wrapper, run on frozen audit storyboards, and
observed to return GPU memory to zero after successful inference. Gemma uses
the same server arguments as the validated former resident audit server and
now follows the same managed lifecycle.

The sk3 profile in `config/sk3_audit_vllm_profiles.json` uses GPU 7 at 42%
memory utilization and the existing shared Qwen cache. It exists for large
shadow-media runs already resident on sk3, avoiding multi-gigabyte cross-node
media transfer. It follows the same batch lease and unload contract. GPU 7 is
outside the production blocklist, and the production batch allocator naturally
prefers a freer allowed GPU while the audit lease is active. A frozen canary
must match the audited sk2 behavior before any corpus-scale invocation.
