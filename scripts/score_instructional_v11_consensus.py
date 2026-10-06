#!/usr/bin/env python3
"""Text-model consensus over two metadata-aware instructional VLM records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SYSTEM = """You are a fail-closed consensus critic. You receive two independent
VLM records for the same storyboard plus fallible metadata proposing a
VIOLATION label. You do not see the storyboard. Never add visual evidence that
neither VLM reports.

Require both VLMs to support the same concrete actor-action-affected-party
event. The proposed violating behavior itself must be performed, not merely
described, prohibited, prevented, cleaned up, repaired, apologized for, or
shown only through consequences. The event polarity must be violation.

The proposed norm must name the same actor's same conduct toward the same
target. Reject labels naming the affected person's response. Sharing, consent,
personal space, turn-taking, sportsmanship, interrupting, apologizing, asking
before borrowing, and littering can be adequately concrete. Bare values such
as respect, fairness, kindness, communication, professionalism, or boundaries
are too broad when a specific act is depicted.

Reject presenters/interviews, isolated characters with no recipient, generic
B-roll/tasks moralized by metadata, and formal safety/legal/medical/technical
procedures. Any actual demo format is allowed.

Return exactly one JSON object with:
records_agree_same_literal_event: yes|no|uncertain
both_support_performed_violation: yes|no|uncertain
both_support_affected_party: yes|no|uncertain
both_support_exact_norm_actor_target: yes|no|uncertain
both_support_violation_polarity: yes|no|uncertain
hard_exclusion: none|presenter_or_description_only|isolated_character_no_interaction|correct_prevention_repair_or_consequence_only|wrong_actor_or_target|generic_or_wrong_norm|formal_or_non_social|visual_disagreement|uncertain
strict_exact_violation: yes|no|uncertain
failure_mechanism: none|visual_disagreement|no_performed_violation|no_affected_party|polarity_mismatch|wrong_actor_or_target|generic_or_wrong_norm|hard_exclusion|uncertain
normalized_literal_event: short actor-action-target description or none
evidence: one short sentence comparing both VLM records to the proposed label

strict_exact_violation=yes only when the first five fields are yes and
hard_exclusion=none. Otherwise fail closed."""


YNU = {"yes", "no", "uncertain"}
EXCLUSIONS = {
    "none",
    "presenter_or_description_only",
    "isolated_character_no_interaction",
    "correct_prevention_repair_or_consequence_only",
    "wrong_actor_or_target",
    "generic_or_wrong_norm",
    "formal_or_non_social",
    "visual_disagreement",
    "uncertain",
}
FAILURES = {
    "none",
    "visual_disagreement",
    "no_performed_violation",
    "no_affected_party",
    "polarity_mismatch",
    "wrong_actor_or_target",
    "generic_or_wrong_norm",
    "hard_exclusion",
    "uncertain",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def latest_successful(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("error") is None and row.get("result") is not None:
            result[row["item_id"]] = row
    return result


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def json_object(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response contained no JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def parse_result(
    text: str,
    qwen: dict[str, Any],
    glm: dict[str, Any],
) -> dict[str, Any]:
    result = json_object(text)
    expected = {
        "records_agree_same_literal_event",
        "both_support_performed_violation",
        "both_support_affected_party",
        "both_support_exact_norm_actor_target",
        "both_support_violation_polarity",
        "hard_exclusion",
        "strict_exact_violation",
        "failure_mechanism",
        "normalized_literal_event",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    atomic = [
        "records_agree_same_literal_event",
        "both_support_performed_violation",
        "both_support_affected_party",
        "both_support_exact_norm_actor_target",
        "both_support_violation_polarity",
        "strict_exact_violation",
    ]
    for key in atomic:
        if result[key] not in YNU:
            raise ValueError(f"invalid {key}: {result[key]!r}")
    repairs: list[str] = []
    for key, allowed in (
        ("hard_exclusion", EXCLUSIONS),
        ("failure_mechanism", FAILURES),
    ):
        if result[key] not in allowed:
            result[f"{key}_raw"] = result[key]
            result[key] = "uncertain"
            repairs.append(key)
    for key in ("normalized_literal_event", "evidence"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{key} must be a nonempty string")

    inputs_strict = all(
        record.get("exact_violation_demo") == "yes"
        and record.get("performed_violation_not_only_described") == "yes"
        and record.get("proposed_actor_behavior_target_match") == "yes"
        and record.get("event_polarity") == "violation"
        for record in (qwen, glm)
    )
    consensus_strict = (
        inputs_strict
        and all(result[key] == "yes" for key in atomic[:5])
        and result["hard_exclusion"] == "none"
        and result["failure_mechanism"] == "none"
    )
    if result["strict_exact_violation"] == "yes" and not consensus_strict:
        result["strict_exact_violation_raw"] = "yes"
        result["strict_exact_violation"] = "uncertain"
        repairs.append("inconsistent_positive")
    if repairs:
        result["strict_exact_violation"] = "uncertain"
        result["validation_repairs"] = sorted(set(repairs))
    return result


def completed_ids(path: Path, model: str) -> set[str]:
    if not path.exists():
        return set()
    return {
        row["item_id"]
        for row in load_jsonl(path)
        if row.get("model") == model
        and row.get("error") is None
        and row.get("result") is not None
    }


def request_one(
    endpoint: str,
    model: str,
    metadata: dict[str, Any],
    qwen_record: dict[str, Any],
    glm_record: dict[str, Any],
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    qwen = qwen_record["result"]
    glm = glm_record["result"]
    packet = {
        "qwen_vlm_record": qwen,
        "glm_vlm_record": glm,
        "fallible_metadata": {
            "title": metadata.get("title"),
            "category": metadata.get("category"),
            "proposed_norm": metadata.get("norm"),
            "proposed_polarity": metadata.get("polarity"),
            "start_quote": metadata.get("start_quote"),
            "end_quote": metadata.get("end_quote"),
            "explanation": metadata.get("explanation"),
        },
    }
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 600,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    "Critique this packet. Return only the required JSON.\n"
                    + json.dumps(packet, ensure_ascii=False, sort_keys=True)
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
                "rubric": "v11_consensus",
                "model": model,
                "qwen_record_sha256": canonical_sha256(qwen),
                "glm_record_sha256": canonical_sha256(glm),
                "result": parse_result(content, qwen, glm),
                "raw_response": content,
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
        "item_id": metadata["item_id"],
        "uid": metadata["uid"],
        "pillar": "instructional",
        "rubric": "v11_consensus",
        "model": model,
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--glm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    metadata_rows = load_jsonl(args.metadata)
    qwen = latest_successful(args.qwen)
    glm = latest_successful(args.glm)
    expected = {row["item_id"] for row in metadata_rows}
    if set(qwen) != expected or set(glm) != expected:
        raise ValueError(
            "VLM coverage mismatch: "
            f"qwen_missing={len(expected - set(qwen))}, "
            f"glm_missing={len(expected - set(glm))}"
        )
    completed = completed_ids(args.out, args.model)
    pending = [row for row in metadata_rows if row["item_id"] not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]
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
                qwen[row["item_id"]],
                glm[row["item_id"]],
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
