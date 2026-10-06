#!/usr/bin/env python3
"""Compare a frozen label-blind episode extraction with an instructional label."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from scripts.run_open_vlm_v10_storyboards import _json_object, load_jsonl, successful_keys
    from scripts.run_instructional_atomic_critic_v1 import (
        SOCIAL_BASES, relation_matches_polarity,
    )
except ModuleNotFoundError:
    from run_open_vlm_v10_storyboards import _json_object, load_jsonl, successful_keys  # type: ignore[no-redef]
    from run_instructional_atomic_critic_v1 import SOCIAL_BASES, relation_matches_polarity  # type: ignore[no-redef]


SYSTEM_SYMBOLIC = """You compare a frozen, label-blind episode extraction to a
proposed weak label. You do not receive pixels. Treat the extraction as the
only evidence of what occurs; never invent a participant, act, setting, or
visual fact from the proposed label.

State the proposed rule as concrete desired or prohibited actor-action-target
conduct. behavior_match=yes only if the frozen observed episode contains that
same behavior, not merely the same topic, setting, object, outcome, or advice.

Choose social_basis operationally: interpersonal_treatment for how a person
treats another; relationship_duty for conduct owed within a relationship;
institutional_fairness for treatment by a school, workplace, business, or
authority; public_shared_conduct for action affecting a community, public, or
shared environment; care_or_welfare for protecting another person's rights,
autonomy, safety, or welfare. self_only, technical_procedure, and none do not
qualify. A viewer, hypothetical person, implied off-screen person, or technical
object is not an affected party.

Determine relation from the frozen observation, not the metadata polarity:
complies follows the rule; violates breaks it; contrast requires both sides or
an explicit correction; illustrates is only for a demonstrated relevant
episode whose direction cannot be reduced further. Do not turn a compliant act
into a violation because the proposed polarity says violation.

Return exactly one JSON object with:
observed_behavior: short actor-action-target restatement from frozen evidence
claimed_rule: short desired/prohibited actor-action-target rule
behavior_match: yes|partial|no|unclear
social_basis: interpersonal_treatment|relationship_duty|institutional_fairness|public_shared_conduct|care_or_welfare|self_only|technical_procedure|none|unclear
relation: complies|violates|contrast|illustrates|unrelated|unclear
evidence: one short comparison sentence

Keep free text under 35 words. Return JSON only."""


TRI_MATCH = {"yes", "partial", "no", "unclear"}
BASES = set(SOCIAL_BASES) | {"self_only", "technical_procedure", "none", "unclear"}
RELATIONS = {"complies", "violates", "contrast", "illustrates", "unrelated", "unclear"}


def derive_label_pass(result: dict[str, Any], episode_pass: bool, polarity: str) -> bool:
    return (
        episode_pass
        and result["behavior_match"] == "yes"
        and result["social_basis"] in SOCIAL_BASES
        and relation_matches_polarity(result["relation"], polarity)
    )


def parse_symbolic(text: str, episode_pass: bool, polarity: str) -> dict[str, Any]:
    result = _json_object(text)
    expected = {"observed_behavior", "claimed_rule", "behavior_match", "social_basis", "relation", "evidence"}
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    for field in ("observed_behavior", "claimed_rule", "evidence"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"invalid {field}")
    if result["behavior_match"] not in TRI_MATCH:
        raise ValueError("invalid behavior_match")
    if result["social_basis"] not in BASES:
        raise ValueError("invalid social_basis")
    if result["relation"] not in RELATIONS:
        raise ValueError("invalid relation")
    result["relation_matches_proposed_polarity"] = relation_matches_polarity(result["relation"], polarity)
    result["label_pass"] = derive_label_pass(result, episode_pass, polarity)
    return result


def latest_success(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        if row.get("error") in (None, "") and isinstance(row.get("result"), dict):
            output[str(row["item_id"])] = row
    return output


def semantic_context(row: dict[str, Any]) -> dict[str, str]:
    return {
        "norm": str(row.get("norm") or ""), "polarity": str(row.get("polarity") or ""),
        "explanation": str(row.get("explanation") or ""),
        "start_quote": str(row.get("start_quote") or ""), "end_quote": str(row.get("end_quote") or ""),
    }


def request_one(endpoint: str, model: str, item_id: str, episode: dict[str, Any], context: dict[str, str], timeout: int, retries: int) -> dict[str, Any]:
    frozen = episode["result"]
    input_value = {
        "frozen_label_blind_episode": frozen,
        "proposed_weak_label": context,
    }
    payload = {
        "model": model, "temperature": 0, "max_tokens": 650,
        "messages": [
            {"role": "system", "content": SYSTEM_SYMBOLIC},
            {"role": "user", "content": "Compare these records:\n" + json.dumps(input_value, sort_keys=True)},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode(); last_error = ""; last_content = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=encoded, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            content = body["choices"][0]["message"].get("content")
            if not content:
                raise ValueError("empty response")
            last_content = content
            return {
                "item_id": item_id, "uid": episode.get("uid"), "pillar": "instructional",
                "model": model, "rubric": "symbolic_label_v1",
                "source_episode_model": episode.get("model"),
                "frozen_episode": frozen, "semantic_context": context,
                "result": parse_symbolic(content, bool(frozen["episode_pass"]), context["polarity"]),
                "raw_response": content, "usage": body.get("usage"), "error": None,
            }
        except urllib.error.HTTPError as exc:
            last_error = f"HTTPError {exc.code}: {exc.read().decode(errors='replace')[:2000]}"
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(2**attempt)
    return {
        "item_id": item_id, "uid": episode.get("uid"), "pillar": "instructional",
        "model": model, "rubric": "symbolic_label_v1", "source_episode_model": episode.get("model"),
        "frozen_episode": frozen, "semantic_context": context, "result": None,
        "raw_response": last_content, "usage": None, "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-output", type=Path, required=True); parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--endpoint", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--timeout", type=int, default=300); parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    episodes = latest_success(load_jsonl(args.episode_output))
    semantics = {str(row["item_id"]): semantic_context(row) for row in load_jsonl(args.semantic_manifest)}
    if set(episodes) != set(semantics):
        raise ValueError("episode and semantic cohorts differ")
    completed = successful_keys(load_jsonl(args.out)) if args.out.exists() else set()
    pending = [item for item in semantics if (item, args.model) not in completed]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(request_one, args.endpoint, args.model, item, episodes[item], semantics[item], args.timeout, args.retries): item
            for item in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result(); handle.write(json.dumps(record, sort_keys=True) + "\n"); handle.flush()
            print(f"{index}/{len(futures)} {record['item_id']} error={record['error']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
