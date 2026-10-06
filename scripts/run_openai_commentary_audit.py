#!/usr/bin/env python3
"""Audit commentary weak labels with structured GPT text output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any


PROMPT_VERSION = "commentary_v2"
SYSTEM_PROMPT = """You audit commentary candidates for weak supervision of social norms.
Use only the detector quote and timestamped transcript context as evidence. The title,
search query, category, agent metadata, detector signal, and proposed norm are fallible
retrieval hints and cannot supply missing evidence.

An accepted item must ground: (1) a specific social actor, which may be a person,
human group, or institution; (2) a concrete behavior; (3) the affected person or
shared social context; and (4) an explicit normative stance such as criticism,
praise, a rule, or a recommended alternative. The behavior and stance may appear in
one sentence or separate nearby transcript spans, but both evidence quotes must be
returned verbatim from the provided detector quote/transcript context.

Reject generic surprise, confusion, outrage, profanity, pain, or excitement; a bare
insult; a behavior description with no judgment; a moral opinion with no concrete
conduct; animal behavior; technical/product procedures; accidents, disasters, or
sports commentary unless a concrete human social behavior is explicitly judged.
Do not infer unstated actors, targets, actions, or norms. Use uncertain rather than
guessing when transcript evidence is incomplete or ambiguous. Use
accept_after_relabel only when all social-behavior and normative-stance evidence is
fully grounded but the proposed norm is wrong, malformed, or too vague; supply a
specific normalized norm in that case."""

TRI = ["yes", "no", "uncertain"]
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_social_norm": {"type": "string", "enum": TRI},
        "concrete_behavior": {"type": "string", "enum": TRI},
        "social_actor_grounded": {"type": "string", "enum": TRI},
        "target_or_shared_context_grounded": {"type": "string", "enum": TRI},
        "normative_stance_grounded": {"type": "string", "enum": TRI},
        "proposed_norm_supported": {"type": "string", "enum": TRI},
        "stance_type": {
            "type": "string",
            "enum": ["criticism", "praise", "rule", "recommended_alternative", "none", "uncertain"],
        },
        "quote_relation": {
            "type": "string",
            "enum": ["same_sentence", "separate_context", "behavior_only", "stance_only", "neither", "uncertain"],
        },
        "decision": {
            "type": "string",
            "enum": ["accept", "accept_after_relabel", "reject", "uncertain"],
        },
        "rejection_reasons": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "generic_affect", "surprise_or_confusion", "bare_insult",
                    "behavior_without_stance", "stance_without_behavior", "actor_missing",
                    "target_context_missing", "nonhuman_behavior", "accident_or_disaster",
                    "sports", "technical_or_product", "off_topic", "title_only_inference",
                    "proposed_norm_mismatch", "insufficient_context", "other",
                ],
            },
        },
        "normalized_behavior": {"type": "string"},
        "normalized_norm": {"type": "string"},
        "behavior_evidence_quote": {"type": "string"},
        "stance_evidence_quote": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": [
        "is_social_norm", "concrete_behavior", "social_actor_grounded",
        "target_or_shared_context_grounded", "normative_stance_grounded",
        "proposed_norm_supported", "stance_type", "quote_relation", "decision",
        "rejection_reasons", "normalized_behavior", "normalized_norm",
        "behavior_evidence_quote", "stance_evidence_quote", "description",
    ],
    "additionalProperties": False,
}


def prompt_hash() -> str:
    value = {"version": PROMPT_VERSION, "system": SYSTEM_PROMPT, "schema": SCHEMA}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def evidence_corpus(item: dict[str, Any]) -> str:
    parts = [str(item.get("start_quote") or "")]
    parts.extend(str(segment.get("text") or "") for segment in item.get("transcript_context") or [])
    return normalized_text(" ".join(parts))


def content_for_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    context = {
        "title_retrieval_hint_only": item.get("title"),
        "category_retrieval_hint_only": item.get("category"),
        "query_retrieval_hint_only": item.get("found_by_query"),
        "agent_metadata_hint_only": item.get("agent"),
        "detector_signal": item.get("polarity"),
        "proposed_norm": item.get("norm"),
        "detector_quote": item.get("start_quote"),
        "detector_start_sec": item.get("start_sec"),
        "detector_end_sec": item.get("end_sec"),
        "transcript_context": item.get("transcript_context") or [],
    }
    return [{"type": "input_text", "text": "Candidate context:\n" + json.dumps(context, ensure_ascii=False)}]


def validate_result(result: dict[str, Any], item: dict[str, Any]) -> None:
    missing = set(SCHEMA["required"]) - set(result)
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    if result["decision"] in {"accept", "accept_after_relabel"}:
        required_yes = (
            "is_social_norm", "concrete_behavior", "social_actor_grounded",
            "target_or_shared_context_grounded", "normative_stance_grounded",
        )
        if any(result[key] != "yes" for key in required_yes):
            raise ValueError("accepted commentary item violates grounding invariants")
        if result["stance_type"] in {"none", "uncertain"}:
            raise ValueError("accepted commentary item has no resolved stance")
        if result["quote_relation"] not in {"same_sentence", "separate_context"}:
            raise ValueError("accepted commentary item does not contain behavior and stance")
        corpus = evidence_corpus(item)
        for key in ("behavior_evidence_quote", "stance_evidence_quote"):
            quote = normalized_text(result[key])
            if not quote or quote not in corpus:
                raise ValueError(f"{key} is not grounded in supplied transcript")
        if not result["normalized_behavior"].strip() or not result["normalized_norm"].strip():
            raise ValueError("accepted commentary item requires normalized behavior and norm")
    if result["decision"] == "accept" and result["proposed_norm_supported"] != "yes":
        raise ValueError("clean acceptance requires proposed_norm_supported=yes")
    if result["decision"] == "accept_after_relabel" and result["proposed_norm_supported"] == "yes":
        raise ValueError("accept_after_relabel requires a deficient proposed norm")
    if result["decision"] == "reject" and not result["rejection_reasons"]:
        raise ValueError("rejected commentary item requires a reason")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.batch / "manifest.json").read_text())
    if manifest.get("pillar") != "commentary" or manifest.get("rubric_version") != PROMPT_VERSION:
        raise SystemExit("batch pillar/rubric does not match commentary_v2")
    items = manifest["items"][: args.limit or None]
    out = args.out or (args.batch / "results.jsonl")
    done = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["item_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    if args.dry_run:
        for item in items:
            if not content_for_item(item):
                raise AssertionError("empty payload")
        print(json.dumps({"items": len(items), "prompt_sha256": prompt_hash()}))
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required; no calls were made")

    from openai import OpenAI

    client = OpenAI()
    with out.open("a") as handle:
        for item in items:
            if item["item_id"] in done:
                continue
            response = client.responses.create(
                model=args.model,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "user", "content": content_for_item(item)}],
                reasoning={"effort": "medium"},
                text={"format": {"type": "json_schema", "name": "commentary_audit", "strict": True, "schema": SCHEMA}},
                max_output_tokens=1100,
                store=False,
            )
            result = json.loads(response.output_text)
            validate_result(result, item)
            usage = response.usage
            record = {
                "item_id": item["item_id"],
                "batch_id": manifest["batch_id"],
                "rubric_version": PROMPT_VERSION,
                "model": args.model,
                "prompt_sha256": prompt_hash(),
                "content_manifest_sha256": item["content_manifest_sha256"],
                "pass_index": 0,
                **result,
                "raw_response": result,
                "response_id": response.id,
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "audited_at": time.time(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
