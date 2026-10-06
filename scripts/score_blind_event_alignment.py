#!/usr/bin/env python3
"""Align two label-blind video observations with a proposed norm.

The video observers never see the label.  This text-only pass may compare their
literal observations with metadata, but it cannot add a visual fact that neither
observer reported.  Results are append-only and safe to resume.
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


RUBRIC_VERSION = "blind_event_alignment_v1"
RUBRIC_VERSION_V2 = "blind_event_alignment_v2"

SYSTEM = """You align two independent, label-blind descriptions of the same
video with a fallible proposed social-norm label.

First find the literal intersection of the observers' evidence. Do not merge
different events and do not introduce an action, target, utterance, motive, or
outcome absent from either observer. A relevant setting is not the behavior.
Objects and occupations do not demonstrate abstract values: sorting objects
does not by itself show acceptance, inclusion, honesty, or respect.

The metadata can help name a behavior only after both observers independently
describe the necessary actor-action-target/shared-setting event. Metadata cannot
rescue a missing reply, handshake, transfer, insult, refusal, helping act, or
other off-screen conduct. If an event is visible but the supplied label is
wrong, preserve it as usable_after_relabel and give a literal normalized label.

Return exactly one JSON object:
both_observers_found_event: yes|no|uncertain
observers_describe_same_event: yes|no|uncertain
intersecting_actor: short phrase or none
intersecting_action: short literal phrase or none
intersecting_target_or_setting: short phrase or none
proposed_norm_supported_by_intersection: yes|no|uncertain
exact_label_supported: yes|no|uncertain
usable_after_relabel: yes|no|uncertain
normalized_behavior: concise actor-neutral behavior or none
visual_evidence_independent_of_metadata: yes|no|uncertain
failure_reason: none|no_event|observer_disagreement|action_mismatch|target_missing|label_mismatch|metadata_only|uncertain
reason: one short sentence"""

SYSTEM_V2 = """You align two independent, label-blind descriptions of the same
video window with a fallible proposed social-norm label. Work only from the
literal intersection of the two observers. Do not merge different events or
introduce an action, target, utterance, motive, cause, or outcome absent from
either description.

A recoverable social-norm event must show concrete informal interpersonal or
shared-public conduct that observers can socially approve or disapprove. A
political rally or policy position, formal/legal enforcement, technical
procedure, generic emotion, crowd presence, setting, reaction, or aftermath is
not by itself such an event. Violence, threats, insults, deception, harassment,
discrimination, boundary violations, unfair treatment, public disruption, and
clearly situated prosocial conduct can qualify.

The proposed label must name the intersecting action itself. It does not pass
when it names an emotion, reaction, abstract value, political/formal position,
or an alleged off-screen cause. Footage of people responding to an alleged
attack does not show the attack. A response-only window may be recoverable only
when the response is itself a concrete social-norm behavior and receives a new
literal label.

Metadata may clarify words already reported by both observers, but it cannot
rescue a missing action or target. Preserve a genuine mismatched social scene as
usable_after_relabel and assign a literal behavior label.

Return exactly one JSON object:
both_observers_found_event: yes|no|uncertain
observers_describe_same_event: yes|no|uncertain
intersecting_actor: short phrase or none
intersecting_action: short literal phrase or none
intersecting_target_or_setting: short phrase or none
literal_social_behavior_visible: yes|no|uncertain
social_norm_domain_of_intersection: tacit_interpersonal|shared_public|political_or_formal|technical_or_procedural|emotion_or_reaction_only|off_topic|uncertain
proposed_label_kind: concrete_conduct|emotion_or_reaction|abstract_value|political_or_formal|technical_or_procedural|malformed|uncertain
proposed_label_action_visible: yes|no|uncertain
response_or_aftermath_only_for_proposed_label: yes|no|uncertain
exact_label_supported: yes|no|uncertain
usable_after_relabel: yes|no|uncertain
normalized_behavior: concise actor-neutral literal behavior or none
visual_evidence_independent_of_metadata: yes|no|uncertain
failure_reason: none|no_event|observer_disagreement|non_social_domain|response_or_aftermath_only|action_mismatch|target_missing|abstract_or_emotion_label|political_or_formal|metadata_only|uncertain
reason: one short sentence"""

REQUIRED = {
    "both_observers_found_event",
    "observers_describe_same_event",
    "intersecting_actor",
    "intersecting_action",
    "intersecting_target_or_setting",
    "proposed_norm_supported_by_intersection",
    "exact_label_supported",
    "usable_after_relabel",
    "normalized_behavior",
    "visual_evidence_independent_of_metadata",
    "failure_reason",
    "reason",
}

REQUIRED_V2 = {
    "both_observers_found_event",
    "observers_describe_same_event",
    "intersecting_actor",
    "intersecting_action",
    "intersecting_target_or_setting",
    "literal_social_behavior_visible",
    "social_norm_domain_of_intersection",
    "proposed_label_kind",
    "proposed_label_action_visible",
    "response_or_aftermath_only_for_proposed_label",
    "exact_label_supported",
    "usable_after_relabel",
    "normalized_behavior",
    "visual_evidence_independent_of_metadata",
    "failure_reason",
    "reason",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def successful_by_item(rows: list[dict]) -> dict[str, dict]:
    return {
        row["item_id"]: row
        for row in rows
        if row.get("result") is not None and row.get("error") is None
    }


def successful_keys(rows: list[dict]) -> set[tuple[str, str, str, str, str]]:
    return {
        (
            row["item_id"],
            row["alignment_model"],
            row["observer_a_model"],
            row["observer_b_model"],
            row.get("rubric_version", RUBRIC_VERSION),
        )
        for row in rows
        if row.get("rubric_version") in {RUBRIC_VERSION, RUBRIC_VERSION_V2}
        and row.get("result") is not None
        and row.get("error") is None
    }


def parse_json(text: str, rubric_version: str = RUBRIC_VERSION) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contained no JSON object")
    result = json.loads(text[start : end + 1])
    required = REQUIRED_V2 if rubric_version == RUBRIC_VERSION_V2 else REQUIRED
    missing = required - result.keys()
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    if rubric_version == RUBRIC_VERSION_V2:
        allowed = {
            "social_norm_domain_of_intersection": {
                "tacit_interpersonal",
                "shared_public",
                "political_or_formal",
                "technical_or_procedural",
                "emotion_or_reaction_only",
                "off_topic",
                "uncertain",
            },
            "proposed_label_kind": {
                "concrete_conduct",
                "emotion_or_reaction",
                "abstract_value",
                "political_or_formal",
                "technical_or_procedural",
                "malformed",
                "uncertain",
            },
            "failure_reason": {
                "none",
                "no_event",
                "observer_disagreement",
                "non_social_domain",
                "response_or_aftermath_only",
                "action_mismatch",
                "target_missing",
                "abstract_or_emotion_label",
                "political_or_formal",
                "metadata_only",
                "uncertain",
            },
        }
        for key, values in allowed.items():
            if result[key] not in values:
                raise ValueError(f"invalid v2 {key}: {result[key]!r}")
        ternary_keys = {
            "both_observers_found_event",
            "observers_describe_same_event",
            "literal_social_behavior_visible",
            "proposed_label_action_visible",
            "response_or_aftermath_only_for_proposed_label",
            "exact_label_supported",
            "usable_after_relabel",
            "visual_evidence_independent_of_metadata",
        }
        for key in ternary_keys:
            if result[key] not in {"yes", "no", "uncertain"}:
                raise ValueError(f"invalid v2 {key}: {result[key]!r}")
        if result["exact_label_supported"] == "yes":
            exact_requirements = {
                "both_observers_found_event": "yes",
                "observers_describe_same_event": "yes",
                "literal_social_behavior_visible": "yes",
                "proposed_label_kind": "concrete_conduct",
                "proposed_label_action_visible": "yes",
                "response_or_aftermath_only_for_proposed_label": "no",
                "visual_evidence_independent_of_metadata": "yes",
                "failure_reason": "none",
            }
            for key, expected in exact_requirements.items():
                if result[key] != expected:
                    raise ValueError(
                        "inconsistent v2 exact label: "
                        f"{key}={result[key]!r}, expected {expected!r}"
                    )
            if result["social_norm_domain_of_intersection"] not in {
                "tacit_interpersonal",
                "shared_public",
            }:
                raise ValueError("inconsistent v2 exact label: non-social domain")
        if result["usable_after_relabel"] == "yes":
            recovery_requirements = {
                "both_observers_found_event": "yes",
                "observers_describe_same_event": "yes",
                "literal_social_behavior_visible": "yes",
                "visual_evidence_independent_of_metadata": "yes",
            }
            for key, expected in recovery_requirements.items():
                if result[key] != expected:
                    raise ValueError(
                        "inconsistent v2 recovery: "
                        f"{key}={result[key]!r}, expected {expected!r}"
                    )
            if result["social_norm_domain_of_intersection"] not in {
                "tacit_interpersonal",
                "shared_public",
            }:
                raise ValueError("inconsistent v2 recovery: non-social domain")
    return result


def request_one(
    endpoint: str,
    model: str,
    row: dict,
    observer_a: dict,
    observer_b: dict,
    timeout: int,
    retries: int,
    rubric_version: str = RUBRIC_VERSION,
) -> dict:
    proposed = row.get("norm") or row.get("normalized_behavior") or "unspecified"
    explanation = row.get("explanation") or ""
    quotes = " ".join(
        str(row.get(key) or "")
        for key in ("start_quote", "end_quote", "reaction")
    ).strip()
    prompt = (
        f"Proposed label: {proposed}\n"
        f"Fallible detector explanation: {explanation[:1000]}\n"
        f"Fallible grounded quote/reaction: {quotes[:900]}\n\n"
        "Label-blind observer A:\n"
        f"{json.dumps(observer_a['result'], ensure_ascii=False, sort_keys=True)}\n\n"
        "Label-blind observer B:\n"
        f"{json.dumps(observer_b['result'], ensure_ascii=False, sort_keys=True)}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 500,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_V2
                if rubric_version == RUBRIC_VERSION_V2
                else SYSTEM,
            },
            {"role": "user", "content": prompt},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode()
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
            message = body["choices"][0]["message"]
            content = message.get("content")
            if not content:
                raise ValueError("empty response")
            return {
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": row["pillar"],
                "rubric_version": rubric_version,
                "alignment_model": model,
                "observer_a_model": observer_a["model"],
                "observer_b_model": observer_b["model"],
                "result": parse_json(content, rubric_version),
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
        "item_id": row["item_id"],
        "uid": row["uid"],
        "pillar": row["pillar"],
        "rubric_version": rubric_version,
        "alignment_model": model,
        "observer_a_model": observer_a["model"],
        "observer_b_model": observer_b["model"],
        "result": None,
        "raw_response": None,
        "usage": None,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observer-a", type=Path, required=True)
    parser.add_argument("--observer-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8175/v1")
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--rubric-version",
        choices=(RUBRIC_VERSION, RUBRIC_VERSION_V2),
        default=RUBRIC_VERSION,
    )
    args = parser.parse_args()

    manifest = load_jsonl(args.manifest)
    observer_a = successful_by_item(load_jsonl(args.observer_a))
    observer_b = successful_by_item(load_jsonl(args.observer_b))
    shared = set(observer_a) & set(observer_b)
    rows = [row for row in manifest if row["item_id"] in shared]
    if not rows:
        raise SystemExit("no successful observer pairs overlap the manifest")
    observer_a_model = observer_a[rows[0]["item_id"]]["model"]
    observer_b_model = observer_b[rows[0]["item_id"]]["model"]
    completed = successful_keys(load_jsonl(args.out)) if args.out.exists() else set()
    pending = [
        row
        for row in rows
        if (
            row["item_id"],
            args.model,
            observer_a_model,
            observer_b_model,
            args.rubric_version,
        )
        not in completed
    ]
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
                args.timeout,
                args.retries,
                args.rubric_version,
            ): row["item_id"]
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{index}/{len(futures)} {result['item_id']} "
                f"{'ok' if result['error'] is None else result['error']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
