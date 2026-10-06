#!/usr/bin/env python3
"""Apply an atomic social-demo gate after two label-blind video observers.

The text model classifies inspectable atoms.  Candidate bands are derived in
code, so the model is never asked for an opaque ``is_social_norm`` decision.
Outputs are append-only and safe to resume.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


RUBRIC_VERSION = "instructional_social_gate_v7"

SYSTEM = """You audit an instructional video candidate using:
1. two independent label-blind visual observations;
2. their literal event intersection; and
3. a fallible label and transcript quote from the saved clip.

Classify the atomic evidence. Do not answer a holistic "is social norm"
question. Do not introduce an action, recipient, reply, consequence, motive, or
setting absent from the evidence.

Any depiction medium may qualify: organic footage, acted scenes, film/TV,
animation, puppets, toys, screen scenarios, static illustrated stories, or
clearly labeled B-roll examples. The medium is never a rejection reason.

A demonstrated social event needs:
- a human or anthropomorphic actor;
- a concrete interpersonal action, situated speech act, shared-public action,
  or conventional appearance example;
- an affected person/group or genuinely shared social setting; and
- a grounded interpersonal, coordination, role, or civic expectation.

The transcript may clarify the semantic content of situated dialogue only when
the visual evidence independently shows the participants or shared scene. It
cannot turn a presenter, generic B-roll, aftermath, or off-screen story into a
demo. An illustrated speech bubble needs a depicted recipient/shared scenario;
explanatory words beside a lone diagram are not an interaction. A public act
such as littering may affect a shared setting without a second visible person.
A professional-dress example may be a static/B-roll demonstration when people,
attire, and the relevant social setting are all visible and the grounded text
specifically identifies the appearance rule.

Use these exact values:
actor_kind: person|human_group|anthropomorphic_character|institution|nonhuman_or_object|none|uncertain
behavior_kind: interpersonal_physical_action|situated_speech_act|shared_public_action|conventional_appearance|technical_procedure|audience_presentation|generic_motion_or_speech|private_self_action|none|uncertain
affected_context_kind: visible_person|human_group|shared_social_setting|implicit_or_offscreen_target|private_self|nonhuman_or_object|none|uncertain
expectation_kind: interpersonal_treatment|shared_coordination|conventional_role_obligation|civic_or_institutional_duty|health_or_safety_only|technical_correctness|private_preference|none|uncertain
depiction_support: complete|partial|context_only|explanation_only|none|uncertain
grounded_quote_names_concrete_behavior: yes|no|uncertain
grounded_quote_behavior_occurs_in_depiction: yes|no|uncertain
audience_presentation_only: yes|no|uncertain
technical_or_nonsocial_activity_only: yes|no|uncertain
label_relation: exact|broad|wrong_but_relabelable|unsupported|uncertain
normalized_behavior: concise actor-neutral behavior or none
failure_reason: none|actor_missing|behavior_missing|target_or_shared_context_missing|expectation_not_social|presentation_only|technical_or_nonsocial|quote_not_concrete|quote_action_not_depicted|unlocalized|label_mismatch|uncertain
reason: one short evidence-grounded sentence

Return exactly one JSON object containing all fields."""

REQUIRED = {
    "actor_kind",
    "behavior_kind",
    "affected_context_kind",
    "expectation_kind",
    "depiction_support",
    "grounded_quote_names_concrete_behavior",
    "grounded_quote_behavior_occurs_in_depiction",
    "audience_presentation_only",
    "technical_or_nonsocial_activity_only",
    "label_relation",
    "normalized_behavior",
    "failure_reason",
    "reason",
}

ACTOR_VALUES = {
    "person",
    "human_group",
    "anthropomorphic_character",
    "institution",
    "nonhuman_or_object",
    "none",
    "uncertain",
}
BEHAVIOR_VALUES = {
    "interpersonal_physical_action",
    "situated_speech_act",
    "shared_public_action",
    "conventional_appearance",
    "technical_procedure",
    "audience_presentation",
    "generic_motion_or_speech",
    "private_self_action",
    "none",
    "uncertain",
}
CONTEXT_VALUES = {
    "visible_person",
    "human_group",
    "shared_social_setting",
    "implicit_or_offscreen_target",
    "private_self",
    "nonhuman_or_object",
    "none",
    "uncertain",
}
EXPECTATION_VALUES = {
    "interpersonal_treatment",
    "shared_coordination",
    "conventional_role_obligation",
    "civic_or_institutional_duty",
    "health_or_safety_only",
    "technical_correctness",
    "private_preference",
    "none",
    "uncertain",
}
DEPICTION_VALUES = {
    "complete",
    "partial",
    "context_only",
    "explanation_only",
    "none",
    "uncertain",
}
LABEL_VALUES = {
    "exact",
    "broad",
    "wrong_but_relabelable",
    "unsupported",
    "uncertain",
}
YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def successful_by_item(rows: list[dict]) -> dict[str, dict]:
    return {
        row["item_id"]: row
        for row in rows
        if row.get("result") is not None and row.get("error") is None
    }


def successful_keys(rows: list[dict]) -> set[tuple[str, str]]:
    return {
        (row["item_id"], row["model"])
        for row in rows
        if row.get("rubric_version") == RUBRIC_VERSION
        and row.get("result") is not None
        and row.get("error") is None
    }


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contained no JSON object")
    result = json.loads(text[start : end + 1])
    missing = REQUIRED - result.keys()
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    enum_fields = {
        "actor_kind": ACTOR_VALUES,
        "behavior_kind": BEHAVIOR_VALUES,
        "affected_context_kind": CONTEXT_VALUES,
        "expectation_kind": EXPECTATION_VALUES,
        "depiction_support": DEPICTION_VALUES,
        "grounded_quote_names_concrete_behavior": YES_NO_UNCERTAIN,
        "grounded_quote_behavior_occurs_in_depiction": YES_NO_UNCERTAIN,
        "audience_presentation_only": YES_NO_UNCERTAIN,
        "technical_or_nonsocial_activity_only": YES_NO_UNCERTAIN,
        "label_relation": LABEL_VALUES,
    }
    for field, values in enum_fields.items():
        if result[field] not in values:
            raise ValueError(f"invalid {field}: {result[field]!r}")
    return result


def derive_band(result: dict) -> str:
    actor_grounded = result["actor_kind"] in {
        "person",
        "human_group",
        "anthropomorphic_character",
        "institution",
    }
    social_behavior = result["behavior_kind"] in {
        "interpersonal_physical_action",
        "situated_speech_act",
        "shared_public_action",
        "conventional_appearance",
    }
    social_context = result["affected_context_kind"] in {
        "visible_person",
        "human_group",
        "shared_social_setting",
    }
    social_expectation = result["expectation_kind"] in {
        "interpersonal_treatment",
        "shared_coordination",
        "conventional_role_obligation",
        "civic_or_institutional_duty",
    }
    semantic_grounding = (
        result["grounded_quote_names_concrete_behavior"] == "yes"
        and result["grounded_quote_behavior_occurs_in_depiction"] == "yes"
    )
    exclusions_clear = (
        result["audience_presentation_only"] == "no"
        and result["technical_or_nonsocial_activity_only"] == "no"
    )
    core_atoms = (
        actor_grounded
        and social_behavior
        and social_context
        and social_expectation
        and semantic_grounding
        and exclusions_clear
    )
    if core_atoms and result["depiction_support"] == "complete":
        if result["label_relation"] == "exact":
            return "v7_exact_candidate"
        if result["label_relation"] in {"broad", "wrong_but_relabelable"}:
            return "v7_relabel_candidate"
    if core_atoms and result["depiction_support"] == "partial":
        return "v7_recut_review"
    return "v7_semantic_or_domain_review"


def prompt_for(
    manifest_row: dict,
    observer_a: dict,
    observer_b: dict,
    alignment: dict,
) -> str:
    quotes = " ".join(
        str(manifest_row.get(key) or "")
        for key in ("start_quote", "end_quote")
    ).strip()
    return (
        f"Fallible proposed label: {manifest_row.get('norm') or 'unspecified'}\n"
        f"Grounded quote: {quotes[:1800]}\n"
        f"Detector explanation: {str(manifest_row.get('explanation') or '')[:1200]}\n\n"
        "Label-blind observer A:\n"
        f"{json.dumps(observer_a['result'], ensure_ascii=False, sort_keys=True)}\n\n"
        "Label-blind observer B:\n"
        f"{json.dumps(observer_b['result'], ensure_ascii=False, sort_keys=True)}\n\n"
        "Literal intersection from the prior stage:\n"
        f"{json.dumps(alignment['result'], ensure_ascii=False, sort_keys=True)}"
    )


def request_one(
    endpoint: str,
    model: str,
    row: dict,
    observer_a: dict,
    observer_b: dict,
    alignment: dict,
    timeout: int,
    retries: int,
) -> dict:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 600,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": prompt_for(row, observer_a, observer_b, alignment),
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode()
    last_error = ""
    raw_response = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            raw_response = body["choices"][0]["message"].get("content")
            if not raw_response:
                raise ValueError("empty response")
            result = parse_json(raw_response)
            return {
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": "instructional",
                "rubric_version": RUBRIC_VERSION,
                "model": model,
                "result": result,
                "derived_band": derive_band(result),
                "raw_response": raw_response,
                "usage": body.get("usage"),
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode(errors="replace")[:2000]
            except Exception:
                detail = ""
            last_error = f"HTTPError {exc.code}: {detail}"
        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(2**attempt)
    return {
        "item_id": row["item_id"],
        "uid": row["uid"],
        "pillar": "instructional",
        "rubric_version": RUBRIC_VERSION,
        "model": model,
        "result": None,
        "derived_band": None,
        "raw_response": raw_response,
        "usage": None,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observer-a", type=Path, required=True)
    parser.add_argument("--observer-b", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8175/v1")
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    manifest = load_jsonl(args.manifest)
    observer_a = successful_by_item(load_jsonl(args.observer_a))
    observer_b = successful_by_item(load_jsonl(args.observer_b))
    alignment = successful_by_item(load_jsonl(args.alignment))
    shared = set(observer_a) & set(observer_b) & set(alignment)
    rows = [row for row in manifest if row["item_id"] in shared]
    if not rows:
        raise SystemExit("no successful evidence triples overlap the manifest")

    completed = successful_keys(load_jsonl(args.out)) if args.out.exists() else set()
    pending = [row for row in rows if (row["item_id"], args.model) not in completed]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle, ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = {
            pool.submit(
                request_one,
                args.endpoint,
                args.model,
                row,
                observer_a[row["item_id"]],
                observer_b[row["item_id"]],
                alignment[row["item_id"]],
                args.timeout,
                args.retries,
            ): row["item_id"]
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{index}/{len(futures)} {result['item_id']} "
                f"{result.get('derived_band') or result['error']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
