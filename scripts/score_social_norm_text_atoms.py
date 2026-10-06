#!/usr/bin/env python3
"""Score inspectable social-norm text atoms without making visual claims."""

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


RUBRIC_VERSION = "social_norm_text_atoms_v2"
SYSTEM = """You audit a fallible proposed norm and its aligned transcript text.
You cannot see the video and must never claim that behavior is visible.

Classify inspectable semantic atoms:
- interpersonal conduct includes treatment, communication, consent, exchange,
  care, aggression, discrimination, courtesy, honesty, or obligations between
  people.
- shared-public conduct includes coordination over common space/resources,
  queues, littering, noise, traffic courtesy, and comparable civic conduct.
- role/appearance conventions include a socially grounded role expectation,
  not generic self-improvement.
- belief, identity, religion, ritual, cultural description, private preference,
  technical procedure, product advice, medical/health advice, and generic
  safety are not by themselves tacit interpersonal norms.
- A concrete behavior must name what an actor does or refrains from doing.
  Abstract labels such as respect, awareness, responsibility, equality, safety,
  relationship management, and professionalism are not concrete without a
  grounded action.
- An affected person/group or genuinely shared setting must be grounded in the
  supplied words. Do not invent a recipient or social setting.

Use exactly:
actor_scope: interpersonal_actors|public_group_or_institution|private_self|nonhuman_or_object|missing|uncertain
behavior_scope: interpersonal_conduct|shared_public_conduct|role_or_appearance_convention|technical_or_procedural|health_or_safety_only|belief_identity_or_ritual|preference_or_self_improvement|generic_topic|missing|uncertain
expectation_scope: interpersonal_treatment|shared_coordination|role_obligation|civic_duty|technical_correctness|health_or_safety|private_preference|identity_belief_or_ritual|none|uncertain
concrete_behavior_grounded: yes|no|uncertain
affected_person_or_shared_context_grounded: yes|no|uncertain
assigned_norm_relation: exact|broad|wrong_but_relabelable|unsupported|uncertain
normalized_behavior: concise actor-neutral behavior or "none"
evidence: a short exact phrase from the supplied text or "none"
reason: one short sentence

Return exactly one JSON object with all fields."""

REQUIRED = {
    "actor_scope",
    "behavior_scope",
    "expectation_scope",
    "concrete_behavior_grounded",
    "affected_person_or_shared_context_grounded",
    "assigned_norm_relation",
    "normalized_behavior",
    "evidence",
    "reason",
}
ENUMS = {
    "actor_scope": {
        "interpersonal_actors",
        "public_group_or_institution",
        "private_self",
        "nonhuman_or_object",
        "missing",
        "uncertain",
    },
    "behavior_scope": {
        "interpersonal_conduct",
        "shared_public_conduct",
        "role_or_appearance_convention",
        "technical_or_procedural",
        "health_or_safety_only",
        "belief_identity_or_ritual",
        "preference_or_self_improvement",
        "generic_topic",
        "missing",
        "uncertain",
    },
    "expectation_scope": {
        "interpersonal_treatment",
        "shared_coordination",
        "role_obligation",
        "civic_duty",
        "technical_correctness",
        "health_or_safety",
        "private_preference",
        "identity_belief_or_ritual",
        "none",
        "uncertain",
    },
    "concrete_behavior_grounded": {"yes", "no", "uncertain"},
    "affected_person_or_shared_context_grounded": {"yes", "no", "uncertain"},
    "assigned_norm_relation": {
        "exact",
        "broad",
        "wrong_but_relabelable",
        "unsupported",
        "uncertain",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_result(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contained no JSON object")
    result = json.loads(cleaned[start : end + 1])
    missing = REQUIRED - set(result)
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    for field, values in ENUMS.items():
        if result[field] not in values:
            raise ValueError(f"invalid {field}: {result[field]!r}")
    return result


def derive_social_candidate(result: dict[str, Any]) -> bool:
    return (
        result["actor_scope"]
        in {"interpersonal_actors", "public_group_or_institution"}
        and result["behavior_scope"]
        in {
            "interpersonal_conduct",
            "shared_public_conduct",
            "role_or_appearance_convention",
        }
        and result["expectation_scope"]
        in {
            "interpersonal_treatment",
            "shared_coordination",
            "role_obligation",
            "civic_duty",
        }
        and result["concrete_behavior_grounded"] == "yes"
        and result["affected_person_or_shared_context_grounded"] == "yes"
        and result["assigned_norm_relation"]
        in {"exact", "broad", "wrong_but_relabelable"}
        and result.get("evidence_in_aligned_transcript") is True
    )


def normalize_evidence(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def evidence_is_grounded(evidence: Any, transcript: str) -> bool:
    normalized_evidence = normalize_evidence(str(evidence or ""))
    if normalized_evidence in {"", "none"}:
        return False
    return normalized_evidence in normalize_evidence(transcript)


def aligned_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(segment.get("text") or "").strip()
        for segment in (row.get("aligned_transcript") or [])
        if isinstance(segment, dict) and str(segment.get("text") or "").strip()
    )


def prompt_for(row: dict[str, Any]) -> str:
    return (
        f"Fallible assigned norm: {row.get('norm') or 'none'}\n"
        f"Detector explanation: {row.get('explanation') or 'none'}\n"
        f"Aligned transcript: {aligned_text(row)[:7000] or 'none'}"
    )


def request_one(
    endpoint: str,
    model: str,
    row: dict[str, Any],
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    prompt = prompt_for(row)
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 500,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode()
    raw_response = None
    last_error = ""
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
            result = parse_result(raw_response)
            result["evidence_in_aligned_transcript"] = evidence_is_grounded(
                result["evidence"],
                aligned_text(row),
            )
            return {
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": row["pillar"],
                "rubric_version": RUBRIC_VERSION,
                "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                "model": model,
                "result": result,
                "derived_social_candidate": derive_social_candidate(result),
                "raw_response": raw_response,
                "usage": body.get("usage"),
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode(errors="replace")[:1200]
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
        "pillar": row["pillar"],
        "rubric_version": RUBRIC_VERSION,
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "model": model,
        "result": None,
        "derived_social_candidate": False,
        "raw_response": raw_response,
        "usage": None,
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8273/v1")
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    parser.add_argument(
        "--pillar",
        action="append",
        choices=("instructional", "witnessed", "commentary"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    if args.pillar:
        rows = [row for row in rows if row["pillar"] in set(args.pillar)]
    completed = (
        {
            row["item_id"]
            for row in read_jsonl(args.out)
            if row.get("error") is None
            and isinstance(row.get("result"), dict)
            and row.get("rubric_version") == RUBRIC_VERSION
            and row.get("model") == args.model
        }
        if args.out.exists()
        else set()
    )
    pending = [row for row in rows if row["item_id"] not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    request_one,
                    args.endpoint,
                    args.model,
                    row,
                    args.timeout,
                    args.retries,
                ): row
                for row in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                handle.write(json.dumps(result, sort_keys=True) + "\n")
                handle.flush()
                print(
                    f"{index}/{len(futures)} {result['item_id']} "
                    f"{result['error'] or 'ok'}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
