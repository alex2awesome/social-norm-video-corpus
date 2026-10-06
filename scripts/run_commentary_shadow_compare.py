#!/usr/bin/env python3
"""Run a shadow commentary prompt on a frozen manifest without corpus mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from src.batch_detect import _clamp_messages
from src.llm_detect import LLMDetector


ACCEPTED = {"accept", "accept_after_relabel"}
STANCE_TYPES = {"criticism", "praise", "rule", "recommended_alternative"}


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def detector_field(item: dict[str, Any], name: str) -> Any:
    statement = item.get("detector_statement")
    if not isinstance(statement, dict):
        statement = {}
    aliases = {
        "quote": ("quote", "start_quote"),
        "query": ("query", "found_by_query"),
        "signal": ("signal", "polarity"),
        "norm": ("norm",),
    }
    for key in aliases[name]:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return statement.get(name)


def transcript_context(item: dict[str, Any]) -> list[dict[str, Any]]:
    value = item.get("context")
    if not value:
        value = item.get("transcript_context")
    return value if isinstance(value, list) else []


def evidence_corpus(item: dict[str, Any]) -> str:
    parts = [str(detector_field(item, "quote") or "")]
    parts.extend(str(segment.get("text") or "") for segment in transcript_context(item))
    return normalized_text(" ".join(parts))


def contract_errors(result: dict[str, Any], item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    decision = result.get("decision")
    if decision not in ACCEPTED | {"reject", "uncertain"}:
        errors.append("invalid_decision")
        return errors
    if result.get("same_event") not in {"yes", "no", "uncertain"}:
        errors.append("invalid_same_event")
    if result.get("stance_type") not in STANCE_TYPES | {"none", "uncertain"}:
        errors.append("invalid_stance_type")
    if result.get("proposed_norm_supported") not in {"yes", "no", "uncertain"}:
        errors.append("invalid_proposed_norm_supported")
    if decision in ACCEPTED:
        for field in ("actor", "behavior", "target_or_shared_context", "normalized_norm"):
            if not str(result.get(field) or "").strip():
                errors.append(f"missing_{field}")
        if result.get("same_event") != "yes":
            errors.append("accepted_without_same_event")
        if result.get("stance_type") not in STANCE_TYPES:
            errors.append("accepted_without_explicit_stance")
        corpus = evidence_corpus(item)
        for field in ("behavior_quote", "stance_quote"):
            quote = normalized_text(str(result.get(field) or ""))
            if not quote or quote not in corpus:
                errors.append(f"ungrounded_{field}")
        if decision == "accept" and result.get("proposed_norm_supported") != "yes":
            errors.append("clean_accept_without_supported_norm")
        if decision == "accept_after_relabel" and result.get("proposed_norm_supported") == "yes":
            errors.append("relabel_with_supported_norm")
    elif not str(result.get("reject_reason") or "").strip():
        errors.append("nonaccept_without_reason")
    return errors


def user_prompt(item: dict[str, Any]) -> str:
    payload = {
        "title_retrieval_hint_only": item.get("title"),
        "query_retrieval_hint_only": detector_field(item, "query"),
        "category_retrieval_hint_only": item.get("category"),
        "proposed_norm_retrieval_hint_only": detector_field(item, "norm"),
        "detector_signal_retrieval_hint_only": detector_field(item, "signal"),
        "detector_quote": detector_field(item, "quote"),
        "start_sec": item.get("start_sec"),
        "end_sec": item.get("end_sec"),
        "timestamped_transcript_context": transcript_context(item),
    }
    return "Candidate context:\n" + json.dumps(payload, ensure_ascii=False) + "\nJSON only."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--prompt", type=Path, default=Path("config/commentary_shadow_v3_prompt.txt"))
    parser.add_argument("--shadow-name", default="commentary_shadow_v3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    root = args.project_root.resolve()
    source = load_object(args.manifest)
    items = source.get("items") or []
    if not isinstance(items, list):
        raise ValueError("manifest items must be a list")
    system_prompt = args.prompt.read_text().strip()
    jobs = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt(item)},
        ]
        for item in items
    ]

    cfg = yaml.safe_load((root / "config/settings.yaml").read_text())
    batch_cfg = cfg.get("batch") or {}
    import torch
    from vllm import LLM, SamplingParams

    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gb, total_gb = free_b / 2**30, total_b / 2**30
    need_gb = float(batch_cfg.get("engine_min_gb", 90))
    if free_gb < need_gb:
        raise SystemExit(f"only {free_gb:.1f} GiB free; need {need_gb:.1f}")
    utilization = min(float(batch_cfg.get("gpu_memory_utilization", 0.55)), (free_gb - 6) / total_gb)
    started = time.time()
    llm = LLM(
        model=batch_cfg["model_path"],
        max_model_len=int(batch_cfg.get("max_model_len", 8192)),
        gpu_memory_utilization=utilization,
    )
    params = SamplingParams(temperature=0, max_tokens=900)
    tokenizer = llm.get_tokenizer()
    budget = int(batch_cfg.get("max_model_len", 8192)) - 916
    clamped = sum(int(_clamp_messages(tokenizer, messages, budget)) for messages in jobs)
    outputs = llm.chat(jobs, params)

    records = []
    for item, output in zip(items, outputs):
        raw = output.outputs[0].text
        parsed = LLMDetector._parse(raw) or {}
        errors = contract_errors(parsed, item)
        records.append(
            {
                "item_id": item.get("item_id"),
                "uid": item.get("uid"),
                "statement_index": item.get("statement_index", item.get("item_index")),
                "manual_reference_decision": item.get("manual_reference_decision"),
                "result": parsed,
                "contract_pass": not errors,
                "contract_errors": errors,
                "raw": raw,
            }
        )
    payload = {
        "kind": "commentary_shadow_compare",
        "shadow_name": args.shadow_name,
        "source_batch_id": source.get("batch_id"),
        "model_path": batch_cfg["model_path"],
        "prompt_path": str(args.prompt),
        "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "items": len(items),
        "jobs": len(jobs),
        "clamped_prompts": clamped,
        "elapsed_seconds": time.time() - started,
        "corpus_mutated": False,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("items", "jobs", "clamped_prompts", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
