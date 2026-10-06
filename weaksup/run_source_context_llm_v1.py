#!/usr/bin/env python3
"""Offline vLLM runner for the unified source-context labeling pass.

Loads the configured Llama-3.3-70B-FP8 snapshot on ONE allowed GPU, labels
context packets in batches with the source_context_v1 contract, validates
every result, and appends to a resumable shard.  Parse/validation failures
are recorded with the raw output and never crash the batch.  The model is
loaded once and released on exit (dynamic-lifecycle policy); GPU selection
and memory gating happen in the launch wrapper, not here.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from weaksup.source_context_contract_v1 import (
    CONTRACT_VERSION,
    SYSTEM_PROMPT,
    packet_prompt,
    validate_result,
)

RUNNER_VERSION = "run_source_context_llm_v1"
JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_result(text: str) -> dict[str, Any]:
    match = JSON_BLOCK.search(text)
    if not match:
        raise ValueError("no JSON object in output")
    return validate_result(json.loads(match.group(0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.exists():
        with args.out.open() as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["uid"])
    packets = []
    with args.packets.open() as handle:
        for line in handle:
            if line.strip():
                packet = json.loads(line)
                if packet["uid"] not in done:
                    packets.append(packet)
    if args.limit is not None:
        packets = packets[: args.limit]
    print(f"packets to label: {len(packets)} (already done: {len(done)})", flush=True)
    if not packets:
        return 0

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model_path,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    labeled = failed = 0
    with args.out.open("a") as out:
        for start in range(0, len(packets), args.batch_size):
            batch = packets[start:start + args.batch_size]
            conversations = [
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": packet_prompt(p)}]
                for p in batch
            ]
            outputs = llm.chat(conversations, sampling)
            for packet, output in zip(batch, outputs):
                text = output.outputs[0].text if output.outputs else ""
                row: dict[str, Any] = {
                    "uid": packet["uid"], "pillars": packet.get("pillars"),
                    "platform": packet.get("platform"),
                    "context_available": packet.get("context_available"),
                    "runner_version": RUNNER_VERSION,
                    "contract_version": CONTRACT_VERSION,
                }
                try:
                    row["result"] = parse_result(text)
                    labeled += 1
                except (ValueError, json.JSONDecodeError, KeyError, TypeError) as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    row["raw_output"] = text[:800]
                    failed += 1
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            print(f"progress {min(start + args.batch_size, len(packets))}/{len(packets)} "
                  f"(labeled={labeled} failed={failed})", flush=True)
    print(json.dumps({"runner_version": RUNNER_VERSION, "labeled": labeled,
                      "failed": failed, "already_done": len(done)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
