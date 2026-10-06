# sk2 managed Qwen lanes

These launchers are the versioned local copies of:

- `/lfs/skampere2/0/alexspan/prefix5_lane.sh`
- `/lfs/skampere2/0/alexspan/budgetmatch_lane.sh`

They manage Qwen3-8B as a batch-scoped resource:

1. Acquire the shared `$HOME/.qwen3_8b_batch.lock`. This serializes the two
   lanes so future runs use at most one H200 for this model.
2. Wait up to 24 hours for a GPU with less than 2 GB allocated, polling every
   five minutes. Exit 75 preserves resumable state for a later invocation.
3. Start vLLM at a measured `gpu_memory_utilization=0.35`.
4. Run the complete batch.
5. Terminate the vLLM process group on success, failure, interrupt, or hangup.

All settings can be overridden through environment variables:
`SERIALIZE_QWEN_LANES`, `MODEL_LANE_LOCK`, `GPU_MEMORY_UTILIZATION`,
`GPU_POLL_SECONDS`, `MAX_GPU_WAIT_MINUTES`, and
`MODEL_START_TIMEOUT_SECONDS`.

## 2026-07-29 sizing audit

The original live servers used `gpu_memory_utilization=0.90`, reserving about
130,337 MiB each. Their observed KV peaks were:

- prefix: 109,889 tokens (14.4% of the original 763,120-token cache)
- budget-match: 80,891 tokens (10.6%)

A clean Qwen3-8B launch at 0.35 on sk2 GPU 2 reserved 51,565 MiB and provided
203,216 KV tokens. A 32-request concurrent smoke test returned 32/32 HTTP 200
responses in 3.41 seconds. The cleanup returned GPU 2 to 0 MiB.

The two experiments already running during deployment were not restarted.
Their shells retained the old script inodes and continue to clean up using the
old launch logic. The managed launchers apply on the next invocation.

Pre-deployment remote backups:

- `/lfs/skampere2/0/alexspan/prefix5_lane.sh.20260729_pre_managed_vllm`
- `/lfs/skampere2/0/alexspan/budgetmatch_lane.sh.20260729_pre_managed_vllm`
