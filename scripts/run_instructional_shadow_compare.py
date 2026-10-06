#!/usr/bin/env python3
"""Run current and a named shadow instructional prompt without mutating the corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml

from src.batch_detect import _clamp_messages
from src.llm_detect import SYS_INSTR, InstructionalDetector, LLMDetector, _locate_quote, _wt_tokens


SHADOW_POLARITIES = {"violation", "correct", "contrast"}
SHADOW_EVIDENCE_KINDS = {
    "enacted_dialogue",
    "recorded_social_action",
    "reported_visible_action",
    "screen_social_exchange",
}
MIN_DEMO_SECONDS = 2.0
MAX_DEMO_SECONDS = 30.0
MAX_REPORTED_VISIBLE_SECONDS = 45.0


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def transcript_text(transcript: dict[str, Any], max_chars: int) -> str:
    text = " ".join(
        str(word.get("word") or "").strip() for word in transcript.get("words") or []
    ).strip()
    return text[:max_chars]


def shadow_user_prompt(text: str, title: str | None) -> str:
    title_part = f'Video title (retrieval hint only): "{title}"\n\n' if title else ""
    return (
        title_part
        + f'Transcript:\n"""{text}"""\n\n'
        + "Return JSON with this shape:\n"
        + '{"is_candidate":"yes"|"no",'
        + '"reject_reason":"<short or null>",'
        + '"demos":[{"polarity":"violation"|"correct"|"contrast",'
        + '"norm":"<concrete canonical rule>",'
        + '"behavior":"<actor action target/shared context>",'
        + '"actor":"<explicit actor>",'
        + '"target_or_shared_context":"<explicit target/context>",'
        + '"evidence_kind":"enacted_dialogue"|"recorded_social_action"|'
        + '"reported_visible_action"|"screen_social_exchange",'
        + '"start_quote":"<exact first words of behavior interval>",'
        + '"end_quote":"<exact last words of behavior interval>",'
        + '"label_quote":"<exact words that judge or state the rule>"}]}\n'
        + "JSON only."
    )


def locate_shadow_demos(result: dict[str, Any], transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words = transcript.get("words") or []
    tokens = _wt_tokens(words)
    located = []
    for demo in result.get("demos") or []:
        if not isinstance(demo, dict):
            continue
        start = _locate_quote(str(demo.get("start_quote") or ""), tokens)
        end = _locate_quote(str(demo.get("end_quote") or ""), tokens)
        label = _locate_quote(str(demo.get("label_quote") or ""), tokens)
        located_demo = {
            **demo,
            "start_sec": start[0] if start else None,
            "end_sec": end[1] if end else (start[1] if start else None),
            "start_quote_grounded": start is not None,
            "end_quote_grounded": end is not None,
            "label_quote_grounded": label is not None,
        }
        reasons = shadow_contract_rejection_reasons(located_demo)
        located_demo["shadow_contract_pass"] = not reasons
        located_demo["shadow_contract_rejection_reasons"] = reasons
        located.append(located_demo)
    return located


def shadow_contract_rejection_reasons(demo: dict[str, Any]) -> list[str]:
    """Return deterministic shadow-contract failures without hiding raw outputs."""
    reasons: list[str] = []
    if demo.get("polarity") not in SHADOW_POLARITIES:
        reasons.append("invalid_polarity")
    evidence_kind = demo.get("evidence_kind")
    if evidence_kind not in SHADOW_EVIDENCE_KINDS:
        reasons.append("invalid_evidence_kind")
    for field in ("norm", "behavior", "actor", "target_or_shared_context"):
        if not str(demo.get(field) or "").strip():
            reasons.append(f"missing_{field}")
    for field in ("start_quote_grounded", "end_quote_grounded", "label_quote_grounded"):
        if demo.get(field) is not True:
            reasons.append(field.replace("_grounded", "_ungrounded"))

    start = demo.get("start_sec")
    end = demo.get("end_sec")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        reasons.append("missing_interval")
        return reasons
    duration = end - start
    if duration <= 0:
        reasons.append("nonpositive_duration")
    elif duration < MIN_DEMO_SECONDS:
        reasons.append("insufficient_temporal_context")
    max_seconds = (
        MAX_REPORTED_VISIBLE_SECONDS
        if evidence_kind == "reported_visible_action"
        else MAX_DEMO_SECONDS
    )
    if duration > max_seconds:
        reasons.append("duration_exceeds_single_event_limit")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--prompt", type=Path, default=Path("config/instructional_shadow_v2_prompt.txt"))
    parser.add_argument("--shadow-name", default="shadow_v2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    root = args.project_root.resolve()
    cfg = yaml.safe_load((root / "config/settings.yaml").read_text())
    source = load_json(args.source_manifest)
    uids = sorted({str(item["uid"]) for item in source.get("items") or []})
    current = InstructionalDetector(cfg)
    shadow_system = args.prompt.read_text().strip()
    jobs = []
    records = []
    for uid in uids:
        transcript_path = root / "data/transcripts" / f"{uid}.json"
        transcript = load_json(transcript_path)
        matching = [item for item in source["items"] if str(item["uid"]) == uid]
        title = matching[0].get("title") if matching else None
        current_messages, current_aux = current.prepare(transcript, title)
        text = transcript_text(transcript, current.max_chars)
        shadow_messages = [
            {"role": "system", "content": shadow_system},
            {"role": "user", "content": shadow_user_prompt(text, title)},
        ]
        record = {
            "uid": uid,
            "title": title,
            "source_item_ids": [item["item_id"] for item in matching],
            "transcript_path": str(transcript_path.relative_to(root)),
        }
        records.append(record)
        if current_messages is not None:
            jobs.append((record, "current", current_messages, current_aux, transcript))
        jobs.append((record, args.shadow_name, shadow_messages, None, transcript))

    import torch
    from vllm import LLM, SamplingParams

    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gb, total_gb = free_b / 2**30, total_b / 2**30
    batch_cfg = cfg.get("batch") or {}
    need_gb = float(batch_cfg.get("engine_min_gb", 90))
    if free_gb < need_gb:
        raise SystemExit(f"only {free_gb:.1f} GiB free on assigned GPU; need {need_gb:.1f}")
    utilization = min(
        float(batch_cfg.get("gpu_memory_utilization", 0.55)),
        (free_gb - 6) / total_gb,
    )
    started = time.time()
    llm = LLM(
        model=batch_cfg["model_path"],
        max_model_len=int(batch_cfg.get("max_model_len", 8192)),
        gpu_memory_utilization=utilization,
    )
    output_tokens = int(cfg.get("llm", {}).get("max_tokens", 1600))
    params = SamplingParams(temperature=0, max_tokens=output_tokens)
    tokenizer = llm.get_tokenizer()
    budget = int(batch_cfg.get("max_model_len", 8192)) - output_tokens - 16
    clamped = 0
    for _record, _kind, messages, _aux, _transcript in jobs:
        clamped += int(_clamp_messages(tokenizer, messages, budget))
    outputs = llm.chat([job[2] for job in jobs], params)
    for (record, kind, _messages, aux, transcript), output in zip(jobs, outputs):
        raw = output.outputs[0].text
        parsed = LLMDetector._parse(raw) or {}
        if kind == "current":
            record["current"] = current.finish(parsed, aux)
            record["current_raw"] = raw
        else:
            record[args.shadow_name] = {
                **parsed,
                "demos": locate_shadow_demos(parsed, transcript),
            }
            record[f"{args.shadow_name}_raw"] = raw

    payload = {
        "kind": f"instructional_{args.shadow_name}_compare",
        "shadow_name": args.shadow_name,
        "source_batch_id": source.get("batch_id"),
        "model_path": batch_cfg["model_path"],
        "current_prompt_sha256": hashlib.sha256(SYS_INSTR.encode()).hexdigest(),
        "shadow_prompt_path": str(args.prompt),
        "shadow_prompt_sha256": hashlib.sha256(shadow_system.encode()).hexdigest(),
        "records": records,
        "uids": len(records),
        "jobs": len(jobs),
        "clamped_prompts": clamped,
        "elapsed_seconds": time.time() - started,
        "corpus_mutated": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("uids", "jobs", "clamped_prompts", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
