#!/usr/bin/env python3
"""Run a shadow audit whose social-scope composites are derived in code.

This deliberately leaves the historical pillar runners unchanged.  It can
re-audit a frozen manifest under the atomic contract without mutating the audit
ledger or production corpus.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import run_openai_commentary_audit as commentary
import run_openai_visual_audit as instructional
import run_openai_witnessed_audit as witnessed
from weak_label_contract import ATOMIC_SCOPE_SCHEMA, attach_scope_labels


PROMPT_VERSION = {
    "instructional": "instructional_atomic_v1",
    "witnessed": "witnessed_atomic_v1",
    "commentary": "commentary_atomic_v1",
}

ATOMIC_SCOPE_PROMPT = """

Do not make a holistic is_social_norm judgment. Instead answer these four
atomic classifications from the supplied evidence:

actor_kind:
- person, human_group, institution: an explicitly grounded human social actor;
- nonhuman: an animal, machine, weather, or other nonhuman actor;
- none: the evidence supplies no actor; uncertain: genuinely ambiguous.

behavior_kind:
- physical_action, speech_act, omission, institutional_action: a temporally
  concrete behavior of that kind;
- technical_procedure: operation, repair, software, product, or job technique;
- accident_or_involuntary_event: harm/event without grounded voluntary conduct;
- abstract_topic_or_trait: only a subject, value, emotion, identity, or trait;
- none: no behavior; uncertain: the behavior kind cannot be resolved.

affected_context_kind:
- person or human_group: the explicitly affected human target;
- shared_social_context: a shared public, household, workplace, classroom, civic,
  or community setting/resource affected by the conduct;
- private_self_only: only the actor's private welfare or self-improvement;
- nonhuman_or_object_only: only an animal, machine, product, or object is affected;
- none: no affected target/context; uncertain: genuinely ambiguous.

expectation_kind:
- interpersonal_treatment: how people treat another person/group;
- shared_coordination: conduct needed to coexist or coordinate in a shared setting;
- civic_or_institutional_duty: how an institution should treat or serve people;
- conventional_role_obligation: a socially expected role duty;
- technical_correctness, legal_rule_only, health_or_physical_safety_only,
  personal_preference: the evidence supports only that non-target basis;
- none: no expectation is expressed or supported; uncertain: genuinely ambiguous.

Use only the category whose definition is grounded. Missing evidence is none or
uncertain, never a guessed in-scope category. The program derives
social_actor_grounded, concrete_behavior, target_or_shared_context_grounded, and
is_social_norm from these answers after your response.
"""


def pillar_module(pillar: str):
    return {
        "instructional": instructional,
        "witnessed": witnessed,
        "commentary": commentary,
    }[pillar]


def atomic_schema(pillar: str) -> dict[str, Any]:
    """Replace model-answered composite scope fields with atomic categories."""
    module = pillar_module(pillar)
    schema = copy.deepcopy(module.SCHEMA)
    derived_fields = {"is_social_norm"}
    if pillar == "commentary":
        derived_fields |= {
            "concrete_behavior",
            "social_actor_grounded",
            "target_or_shared_context_grounded",
        }
    for field in derived_fields:
        schema["properties"].pop(field)
    schema["required"] = [
        field for field in schema["required"] if field not in derived_fields
    ]
    schema["properties"].update(copy.deepcopy(ATOMIC_SCOPE_SCHEMA))
    schema["required"] = list(ATOMIC_SCOPE_SCHEMA) + schema["required"]
    return schema


def system_prompt(pillar: str) -> str:
    return pillar_module(pillar).SYSTEM_PROMPT + ATOMIC_SCOPE_PROMPT


def prompt_hash(pillar: str) -> str:
    value = {
        "version": PROMPT_VERSION[pillar],
        "system": system_prompt(pillar),
        "schema": atomic_schema(pillar),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def derive_and_validate(
    pillar: str, raw_result: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    """Attach deterministic composites, then reuse the audited pillar invariant."""
    result = attach_scope_labels(raw_result)
    pillar_module(pillar).validate_result(result, item)
    return result


def content_for_item(
    pillar: str, batch: Path, item: dict[str, Any]
) -> list[dict[str, Any]]:
    module = pillar_module(pillar)
    if pillar == "commentary":
        return module.content_for_item(item)
    return module.content_for_item(batch, item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.dry_run:
        raise SystemExit(f"refusing to overwrite {args.out}")
    manifest = json.loads((args.batch / "manifest.json").read_text())
    pillar = manifest.get("pillar")
    if pillar not in PROMPT_VERSION:
        raise SystemExit(f"unsupported pillar: {pillar!r}")
    items = manifest.get("items") or []
    if pillar != "commentary":
        items = [item for item in items if item.get("frames")]
    if args.limit:
        items = items[: args.limit]

    if args.dry_run:
        payloads = [content_for_item(pillar, args.batch, item) for item in items]
        if any(not payload for payload in payloads):
            raise AssertionError("empty audit payload")
        print(
            json.dumps(
                {
                    "pillar": pillar,
                    "items": len(items),
                    "prompt_version": PROMPT_VERSION[pillar],
                    "prompt_sha256": prompt_hash(pillar),
                    "model_answers_composite_scope": False,
                }
            )
        )
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required; no calls were made")

    from openai import OpenAI

    client = OpenAI()
    schema = atomic_schema(pillar)
    with args.out.open("x") as handle:
        for item in items:
            response = client.responses.create(
                model=args.model,
                instructions=system_prompt(pillar),
                input=[
                    {
                        "role": "user",
                        "content": content_for_item(pillar, args.batch, item),
                    }
                ],
                reasoning={"effort": "medium"},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": f"{pillar}_atomic_audit",
                        "strict": True,
                        "schema": schema,
                    }
                },
                max_output_tokens=1300,
                store=False,
            )
            raw_result = json.loads(response.output_text)
            result = derive_and_validate(pillar, raw_result, item)
            usage = response.usage
            record = {
                "item_id": item["item_id"],
                "batch_id": manifest["batch_id"],
                "source_rubric_version": manifest.get("rubric_version"),
                "rubric_version": PROMPT_VERSION[pillar],
                "model": args.model,
                "prompt_sha256": prompt_hash(pillar),
                "pass_index": 0,
                **result,
                "raw_response": raw_result,
                "response_id": response.id,
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "audited_at": time.time(),
                "corpus_mutated": False,
            }
            if pillar == "commentary":
                record["content_manifest_sha256"] = item["content_manifest_sha256"]
            else:
                record["frame_manifest_sha256"] = item["frame_manifest_sha256"]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
