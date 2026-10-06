#!/usr/bin/env python3
"""Judge exact causal label alignment from a label-blind visual event record."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SYSTEM = """You are the second, text-only stage of a fail-closed social-norm
video audit. You receive:

1. A LABEL-BLIND visual record produced from a dense storyboard. Treat this as
   the only evidence of what is visibly performed. It can be wrong or vague,
   but you may not invent missing visual actions.
2. Fallible metadata proposing a VIOLATION norm plus grounded clip transcript.
   Transcript may disambiguate audible dialogue or narrate an action that the
   visual record independently depicts. It may not manufacture an off-screen
   event.

Decide whether the exact proposed violation is the same visible actor's same
performed behavior toward the same target.

Reject these causal mistakes:
- the label names the affected person's response, a repair, consequence, or
  correct behavior rather than the offense;
- the visible act is accidental but the label implies deliberate disrespect,
  refusal, ignoring, coercion, or intent;
- the record shows generic interaction, ordinary service, or an object task
  and metadata supplies the moral meaning;
- the norm is a broad value such as respect, empathy, self-control,
  professionalism, boundaries, tone, communication, or inclusion when the
  visible record supports only a more specific action;
- a rule, hypothetical, prohibition, or described example is not performed;
- the transcript and visual record describe different actors, objects, or
  actions.

Allowed formats include live action, role-play, animation, illustrated story,
text dialogue, and explicit physical etiquette or gesture demonstrations.

Return exactly one JSON object with:
visual_record_supports_concrete_demo: yes|no|uncertain
proposed_norm_names_literal_action: yes|no|uncertain
same_actor_and_target: yes|no|uncertain
causal_or_intent_match: yes|no|not_required|uncertain
proposed_violation_polarity_matches: yes|no|uncertain
quote_role: performed_dialogue|narrated_depicted_action|description_only|prohibition_or_hypothetical|corrective_response_or_consequence|uncertain
norm_relation: exact|too_broad|different_behavior|wrong_target|wrong_cause|formal_or_non_social|uncertain
strict_exact_alignment: yes|no|uncertain
failure_mechanism: none|no_concrete_visual_demo|only_described_or_hypothetical|polarity_mismatch|wrong_actor_or_target|wrong_cause_or_intent|norm_too_broad|different_behavior|formal_or_non_social|uncertain
normalized_visible_action: one short actor-action-target description or none
evidence: one short sentence contrasting the visual record and metadata

strict_exact_alignment=yes only if the first three fields are yes,
causal_or_intent_match is yes or not_required,
proposed_violation_polarity_matches=yes, quote_role is performed_dialogue or
narrated_depicted_action, norm_relation=exact, and failure_mechanism=none.
Fail closed otherwise."""


YNU = {"yes", "no", "uncertain"}
CAUSAL = {"yes", "no", "not_required", "uncertain"}
QUOTE_ROLES = {
    "performed_dialogue",
    "narrated_depicted_action",
    "description_only",
    "prohibition_or_hypothetical",
    "corrective_response_or_consequence",
    "uncertain",
}
NORM_RELATIONS = {
    "exact",
    "too_broad",
    "different_behavior",
    "wrong_target",
    "wrong_cause",
    "formal_or_non_social",
    "uncertain",
}
FAILURES = {
    "none",
    "no_concrete_visual_demo",
    "only_described_or_hypothetical",
    "polarity_mismatch",
    "wrong_actor_or_target",
    "wrong_cause_or_intent",
    "norm_too_broad",
    "different_behavior",
    "formal_or_non_social",
    "uncertain",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_success(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("error") is None and isinstance(row.get("result"), dict):
            latest[row["item_id"]] = row
    return latest


def json_object(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response contained no JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def parse_result(text: str) -> dict[str, Any]:
    result = json_object(text)
    expected = {
        "visual_record_supports_concrete_demo",
        "proposed_norm_names_literal_action",
        "same_actor_and_target",
        "causal_or_intent_match",
        "proposed_violation_polarity_matches",
        "quote_role",
        "norm_relation",
        "strict_exact_alignment",
        "failure_mechanism",
        "normalized_visible_action",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    for key in (
        "visual_record_supports_concrete_demo",
        "proposed_norm_names_literal_action",
        "same_actor_and_target",
        "proposed_violation_polarity_matches",
        "strict_exact_alignment",
    ):
        if result[key] not in YNU:
            raise ValueError(f"invalid {key}: {result[key]!r}")
    if result["causal_or_intent_match"] not in CAUSAL:
        raise ValueError("invalid causal_or_intent_match")
    if result["quote_role"] not in QUOTE_ROLES:
        raise ValueError("invalid quote_role")
    if result["norm_relation"] not in NORM_RELATIONS:
        raise ValueError("invalid norm_relation")
    if result["failure_mechanism"] not in FAILURES:
        raise ValueError("invalid failure_mechanism")
    for key in ("normalized_visible_action", "evidence"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"invalid {key}")
    strict = (
        result["visual_record_supports_concrete_demo"] == "yes"
        and result["proposed_norm_names_literal_action"] == "yes"
        and result["same_actor_and_target"] == "yes"
        and result["causal_or_intent_match"] in {"yes", "not_required"}
        and result["proposed_violation_polarity_matches"] == "yes"
        and result["quote_role"]
        in {"performed_dialogue", "narrated_depicted_action"}
        and result["norm_relation"] == "exact"
        and result["failure_mechanism"] == "none"
    )
    if result["strict_exact_alignment"] == "yes" and not strict:
        result["strict_exact_alignment_raw"] = "yes"
        result["strict_exact_alignment"] = "uncertain"
        result["validation_repair"] = "inconsistent_positive"
    return result


def request_one(
    endpoint: str,
    model: str,
    metadata: dict[str, Any],
    visual_record: dict[str, Any],
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    evidence = {
        "visual_record": visual_record["result"],
        "metadata": {
            key: metadata.get(key)
            for key in (
                "title",
                "category",
                "norm",
                "polarity",
                "start_quote",
                "end_quote",
                "explanation",
            )
        },
    }
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 650,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    "Judge this frozen evidence. Return only the required JSON.\n"
                    + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
                ),
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode()
    last_error = ""
    last_content = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            content = body["choices"][0]["message"].get("content")
            if not content:
                raise ValueError("empty response")
            last_content = content
            return {
                "item_id": metadata["item_id"],
                "uid": metadata["uid"],
                "pillar": "instructional",
                "rubric": "v12_causal_alignment",
                "model": model,
                "visual_model": visual_record.get("model"),
                "result": parse_result(content),
                "raw_response": content,
                "usage": body.get("usage"),
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            last_error = f"HTTPError {exc.code}: {detail}"
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(2**attempt)
    return {
        "item_id": metadata["item_id"],
        "uid": metadata["uid"],
        "pillar": "instructional",
        "rubric": "v12_causal_alignment",
        "model": model,
        "visual_model": visual_record.get("model"),
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--visual-records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    metadata = load_jsonl(args.metadata)
    visual = latest_success(load_jsonl(args.visual_records))
    previous = load_jsonl(args.out) if args.out.exists() else []
    complete = set(latest_success(previous))
    pending = [
        row
        for row in metadata
        if row["item_id"] in visual and row["item_id"] not in complete
    ]
    if len(visual) < len(metadata):
        missing = [row["item_id"] for row in metadata if row["item_id"] not in visual]
        raise ValueError(f"missing visual records: {missing[:5]}")
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
                visual[row["item_id"]],
                args.timeout,
                args.retries,
            ): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            result = future.result()
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{index}/{len(pending)} {row['item_id']} "
                f"{'ok' if result['error'] is None else result['error']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
