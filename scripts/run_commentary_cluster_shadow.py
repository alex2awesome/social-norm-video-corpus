#!/usr/bin/env python3
"""Cluster reviewed commentary positives by source event using a shadow LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.batch_detect import _clamp_messages
from src.llm_detect import LLMDetector


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("JSONL rows must be objects")
            rows.append(value)
    return rows


def source_uid(item_id: str) -> str:
    parts = item_id.split(":")
    if len(parts) < 3 or parts[0] != "commentary":
        raise ValueError(f"invalid commentary item_id: {item_id}")
    return ":".join(parts[1:-1])


def validate_clusters(
    input_rows: list[dict[str, Any]], clusters: Any
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    expected = {str(row["item_id"]) for row in input_rows}
    uid_by_id = {item_id: source_uid(item_id) for item_id in expected}
    if not isinstance(clusters, list):
        return False, ["clusters_not_a_list"]
    seen: list[str] = []
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            errors.append(f"cluster_{index}_not_an_object")
            continue
        members = cluster.get("member_item_ids")
        representative = cluster.get("representative_item_id")
        if not isinstance(members, list) or not members:
            errors.append(f"cluster_{index}_has_no_members")
            continue
        members = [str(value) for value in members]
        seen.extend(members)
        if representative not in members:
            errors.append(f"cluster_{index}_representative_not_a_member")
        unknown = set(members) - expected
        if unknown:
            errors.append(f"cluster_{index}_unknown_members={sorted(unknown)}")
        member_uids = {uid_by_id[value] for value in members if value in uid_by_id}
        if len(member_uids) > 1:
            errors.append(f"cluster_{index}_crosses_source_videos")
    duplicates = sorted({item_id for item_id in seen if seen.count(item_id) > 1})
    if duplicates:
        errors.append(f"duplicate_members={duplicates}")
    missing = sorted(expected - set(seen))
    if missing:
        errors.append(f"missing_members={missing}")
    return not errors, errors


def prompt_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "item_id", "normalized_behavior", "normalized_norm",
        "behavior_evidence_quote", "stance_evidence_quote",
    )
    return [{field: row.get(field) for field in fields} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--prompt", type=Path, default=Path("config/commentary_cluster_shadow_v1_prompt.txt"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    root = args.project_root.resolve()
    cfg = yaml.safe_load((root / "config/settings.yaml").read_text())
    accepted = [
        row for row in load_jsonl(args.results)
        if str(row.get("decision") or "").startswith("accept")
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        grouped[source_uid(str(row["item_id"]))].append(row)
    system_prompt = args.prompt.read_text().strip()
    jobs = []
    for uid, rows in sorted(grouped.items()):
        jobs.append((uid, rows, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Accepted statements:\n" + json.dumps(prompt_items(rows), ensure_ascii=False) + "\nJSON only."},
        ]))

    import torch
    from vllm import LLM, SamplingParams

    batch_cfg = cfg.get("batch") or {}
    free_b, total_b = torch.cuda.mem_get_info(0)
    need_gb = float(batch_cfg.get("engine_min_gb", 90))
    free_gb, total_gb = free_b / 2**30, total_b / 2**30
    if free_gb < need_gb:
        raise SystemExit(f"only {free_gb:.1f} GiB free; need {need_gb:.1f}")
    utilization = min(float(batch_cfg.get("gpu_memory_utilization", 0.55)), (free_gb - 6) / total_gb)
    started = time.time()
    llm = LLM(model=batch_cfg["model_path"], max_model_len=int(batch_cfg.get("max_model_len", 8192)), gpu_memory_utilization=utilization)
    params = SamplingParams(temperature=0, max_tokens=1200)
    tokenizer = llm.get_tokenizer()
    budget = int(batch_cfg.get("max_model_len", 8192)) - 1216
    clamped = sum(int(_clamp_messages(tokenizer, messages, budget)) for _, _, messages in jobs)
    outputs = llm.chat([messages for _, _, messages in jobs], params)
    records = []
    for (uid, rows, _messages), output in zip(jobs, outputs):
        raw = output.outputs[0].text
        parsed = LLMDetector._parse(raw) or {}
        clusters = parsed.get("clusters")
        valid, errors = validate_clusters(rows, clusters)
        records.append({
            "uid": uid,
            "input_item_ids": [row["item_id"] for row in rows],
            "clusters": clusters,
            "contract_valid": valid,
            "contract_errors": errors,
            "raw": raw,
        })
    payload = {
        "kind": "commentary_event_cluster_shadow_v1",
        "source_results": str(args.results),
        "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "model_path": batch_cfg["model_path"],
        "accepted_inputs": len(accepted),
        "source_videos": len(grouped),
        "jobs": len(jobs),
        "clamped_prompts": clamped,
        "elapsed_seconds": time.time() - started,
        "corpus_mutated": False,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("accepted_inputs", "source_videos", "jobs", "clamped_prompts", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
