#!/usr/bin/env python3
"""Shadow-score whether a proposed label is an operational social norm.

The scorer uses only text metadata and writes append-only JSONL.  It is a
semantic routing feature, never evidence that an event is visible.
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


SYSTEM = """Decide whether a proposed label describes a concrete social norm.
Operational definition: a person performs or omits a recognizable behavior that
affects another person or a shared social setting, and observers could reasonably
approve/disapprove of that conduct. Tacit interpersonal and shared-public
expectations count.

Do NOT count:
- software, appliance, exercise, art, academic, or other technical procedures;
- health, hygiene, safety, or self-improvement instructions without a social target;
- abstract values with no recognizable actor/action/affected party;
- emotions, reactions, outcomes, or personality traits mislabeled as conduct;
- formal/legal rules merely being explained, unless a concrete interpersonal act
  is also named.

This is a text-only routing prior. Do not claim the action is visible.
Return exactly one JSON object:
social_norm_candidate: yes|no|uncertain
concrete_behavior_named: yes|no|uncertain
affected_other_or_shared_setting_named: yes|no|uncertain
norm_type: tacit_interpersonal|shared_public|formal_rule|technical_procedure|health_hygiene|self_regulation|abstract_value|off_topic|uncertain
actor: short phrase or none
action: short phrase or none
affected_party_or_setting: short phrase or none
reason: one short sentence"""

SYSTEM_V2 = """Decide whether the supplied text identifies a concrete social
norm candidate. This is a text-only routing prior; never claim that the act is
visible.

A social norm candidate must resolve to:
1. an actor;
2. a recognizable action, omission, or situated speech act;
3. another person or a genuinely social shared setting affected by it; and
4. an informal interpersonal/shared-public expectation observers could
   reasonably approve or disapprove.

Include discrimination, stereotyping, exclusion, deception, insults, consent,
queueing, littering, disruptive noise, unfair treatment, and situated pledges
or refusals when the explanation/quote makes the concrete conduct clear. A
one-word abstract label such as "equality" or "inclusion" may pass only when
the supplied context resolves it to a specific action and affected group; put
that action in normalized_social_behavior.

Exclude equipment operation, software/appliance use, exercise/sport technique,
art/academic procedures, traffic-operation rules, animal/medical safety,
toileting/dental/self-care, and generic self-improvement. A shared machine,
road, classroom lesson, or workplace topic is not by itself a social setting:
the named behavior must primarily coordinate or affect people. Also exclude
emotions, reactions, outcomes, malformed labels, and abstract values whose
context still lacks a concrete act.

Return exactly one JSON object:
social_norm_candidate: yes|no|uncertain
concrete_behavior_named: yes|no|uncertain
affected_other_or_shared_setting_named: yes|no|uncertain
norm_type: tacit_interpersonal|shared_public|formal_rule|technical_procedure|health_hygiene|self_regulation|abstract_value|off_topic|uncertain
actor: short phrase or none
action: short phrase or none
affected_party_or_setting: short phrase or none
normalized_social_behavior: concise actor-neutral action or none
reason: one short sentence"""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def successful_item_ids(rows: list[dict]) -> set[str]:
    """Only successful rows suppress a retry in an append-only score file."""
    return {
        row["item_id"]
        for row in rows
        if row.get("result") is not None and row.get("error") is None
    }


def parse_json(text: str, rubric: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contained no JSON object")
    result = json.loads(text[start : end + 1])
    required = {
        "social_norm_candidate",
        "concrete_behavior_named",
        "affected_other_or_shared_setting_named",
        "norm_type",
        "actor",
        "action",
        "affected_party_or_setting",
        "reason",
    }
    if rubric == "v2":
        required.add("normalized_social_behavior")
    missing = required - result.keys()
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    return result


def request_one(
    endpoint: str, model: str, row: dict, timeout: int, rubric: str
) -> dict:
    label = row.get("norm") or row.get("normalized_behavior") or ""
    explanation = row.get("explanation") or row.get("reaction_context") or ""
    quote = " ".join(
        str(row.get(key) or "")
        for key in ("start_quote", "end_quote", "reaction")
    ).strip()
    prompt = (
        f"Title: {row.get('title') or ''}\n"
        f"Category: {row.get('category') or ''}\n"
        f"Proposed label: {label}\n"
        f"Detector explanation/context: {str(explanation)[:1600]}\n"
        f"Grounded quote/reaction: {quote[:1200]}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 350,
        "messages": [
            {"role": "system", "content": SYSTEM if rubric == "v1" else SYSTEM_V2},
            {"role": "user", "content": prompt},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
        content = body["choices"][0]["message"].get("content")
        if not content:
            raise ValueError("empty response")
        return {
            "item_id": row["item_id"],
            "uid": row["uid"],
            "pillar": row["pillar"],
            "rubric": rubric,
            "result": parse_json(content, rubric),
            "raw_response": content,
            "usage": body.get("usage"),
            "error": None,
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "item_id": row["item_id"],
            "uid": row["uid"],
            "pillar": row["pillar"],
            "rubric": rubric,
            "result": None,
            "raw_response": None,
            "usage": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8175/v1")
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--rubric", choices=("v1", "v2"), default="v1")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    completed = (
        successful_item_ids(load_jsonl(args.out)) if args.out.exists() else set()
    )
    pending = [row for row in rows if row["item_id"] not in completed]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def score(row: dict) -> dict:
        result = request_one(args.endpoint, args.model, row, args.timeout, args.rubric)
        if args.delay:
            time.sleep(args.delay)
        return result

    with args.out.open("a") as handle, ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = {pool.submit(score, row): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            result = future.result()
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{index}/{len(pending)} {row['item_id']} "
                f"{'ok' if not result['error'] else result['error']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
