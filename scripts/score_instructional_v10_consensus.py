#!/usr/bin/env python3
"""Score consensus between two label-blind visual records and weak metadata."""

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


SYSTEM_V10C = """You are a fail-closed consensus critic for instructional
social-norm video clips. You receive two independent LABEL-BLIND visual records
of the same dense storyboard plus fallible detector metadata. You do not see
the video. Never add an action that neither visual record reports.

The clip is usable only if both visual records support the same concrete
performed social event. Compare their literal actor, action/situated utterance,
affected party/shared setting, and scene role. Generic agreement that "people
talk" is insufficient.

Apply hard exclusions:
- presenter/example speech or retrospective description without the affected
  party in a situated interaction;
- formal safety, traffic, legal, house, medical, service-transfer, technical,
  developmental, diagnostic, motor, or ordinary-task instruction;
- generic B-roll, montage, aftermath, or social meaning supplied by metadata.

Any deliberate demo format is allowed, including animation, text dialogue,
role-play, film, puppets, and explicitly performed etiquette.

Judge the original norm at the granularity useful for weak supervision.
Conventional behavior classes such as sharing, consent, personal space,
turn-taking, sportsmanship, greeting, handshake etiquette, apologizing,
honesty, and asking before borrowing are adequately concrete when they match
the event. Bare values/domains such as "respect", "care", "kindness",
"professionalism", "customer service", "behavior", "communication", or
"boundaries" are generic when the event is actually a specific insult, bias,
fraud, coercion, refusal, or abuse. Also reject labels naming a different act
or affected party.

Polarity must attach to the actor whose behavior the norm names. A person
asking politely is not a violation merely because peers later exclude them.
An explanation of a rule is not the performed event's polarity.

Return one JSON object with exactly:
visual_records_agree_on_literal_event: yes|no|uncertain
performed_demo_supported_by_both: yes|no|uncertain
hard_exclusion: none|presenter_or_description_only|formal_safety_or_legal|medical_service_or_technical_procedure|developmental_or_diagnostic_task|ordinary_action_metadata_moralized|broll_montage_or_aftermath|uncertain
norm_granularity: adequate_conventional_class|generic_value_or_domain|different_behavior|wrong_affected_party|formal_or_non_social|uncertain
polarity_attaches_to_labeled_actor: yes|no|uncertain
strict_exact_candidate: yes|no|uncertain
usable_after_relabel: yes|no|uncertain
safe_relabel: short concrete norm label or none
normalized_performed_behavior: short literal behavior or none
failure_mechanism: none|visual_disagreement|demo_not_supported_by_both|hard_exclusion|generic_label|different_behavior|wrong_affected_party|polarity_target_mismatch|uncertain
evidence: one short sentence comparing both literal records with the proposed label

strict_exact_candidate=yes only when the first two fields are yes,
hard_exclusion=none, norm_granularity=adequate_conventional_class, and polarity
attaches to the labeled actor=yes. Otherwise fail closed."""


YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}
HARD_EXCLUSIONS = {
    "none",
    "presenter_or_description_only",
    "formal_safety_or_legal",
    "medical_service_or_technical_procedure",
    "developmental_or_diagnostic_task",
    "ordinary_action_metadata_moralized",
    "broll_montage_or_aftermath",
    "uncertain",
}
NORM_GRANULARITIES = {
    "adequate_conventional_class",
    "generic_value_or_domain",
    "different_behavior",
    "wrong_affected_party",
    "formal_or_non_social",
    "uncertain",
}
FAILURES = {
    "none",
    "visual_disagreement",
    "demo_not_supported_by_both",
    "hard_exclusion",
    "generic_label",
    "different_behavior",
    "wrong_affected_party",
    "polarity_target_mismatch",
    "uncertain",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def latest_successful(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("error") is None and row.get("result") is not None:
            latest[row["item_id"]] = row
    return latest


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected one JSON object")
    return value


def parse_v10c(
    text: str,
    qwen_visual: dict[str, Any],
    glm_visual: dict[str, Any],
) -> dict[str, Any]:
    result = _json_object(text)
    expected = {
        "visual_records_agree_on_literal_event",
        "performed_demo_supported_by_both",
        "hard_exclusion",
        "norm_granularity",
        "polarity_attaches_to_labeled_actor",
        "strict_exact_candidate",
        "usable_after_relabel",
        "safe_relabel",
        "normalized_performed_behavior",
        "failure_mechanism",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    atomic = [
        "visual_records_agree_on_literal_event",
        "performed_demo_supported_by_both",
        "polarity_attaches_to_labeled_actor",
        "strict_exact_candidate",
        "usable_after_relabel",
    ]
    for key in atomic:
        if result[key] not in YES_NO_UNCERTAIN:
            raise ValueError(f"invalid {key}: {result[key]!r}")
    enum_specs = {
        "hard_exclusion": (HARD_EXCLUSIONS, "uncertain"),
        "norm_granularity": (NORM_GRANULARITIES, "uncertain"),
        "failure_mechanism": (FAILURES, "uncertain"),
    }
    enum_repairs: list[str] = []
    for key, (allowed, fallback) in enum_specs.items():
        if result[key] not in allowed:
            result[f"{key}_raw"] = result[key]
            result[key] = fallback
            enum_repairs.append(key)
    string_repairs: list[str] = []
    for key in ["safe_relabel", "normalized_performed_behavior", "evidence"]:
        if not isinstance(result[key], str) or not result[key].strip():
            result[f"{key}_raw"] = result[key]
            result[key] = "none"
            string_repairs.append(key)

    qwen_pass = (
        qwen_visual.get("observable_event") == "yes"
        and qwen_visual.get("same_event_actor_action_target") == "yes"
        and qwen_visual.get("event_temporally_localized") == "yes"
        and qwen_visual.get("scene_role") == "demonstrated_event"
    )
    glm_pass = glm_visual.get("demo_usable") == "yes"
    strict_consistent = (
        qwen_pass
        and glm_pass
        and result["visual_records_agree_on_literal_event"] == "yes"
        and result["performed_demo_supported_by_both"] == "yes"
        and result["hard_exclusion"] == "none"
        and result["norm_granularity"] == "adequate_conventional_class"
        and result["polarity_attaches_to_labeled_actor"] == "yes"
    )
    relabel_consistent = (
        qwen_pass
        and glm_pass
        and result["visual_records_agree_on_literal_event"] == "yes"
        and result["performed_demo_supported_by_both"] == "yes"
        and result["hard_exclusion"] == "none"
        and result["usable_after_relabel"] == "yes"
        and result["safe_relabel"].lower() != "none"
    )
    repairs: list[str] = []
    if result["strict_exact_candidate"] == "yes" and not strict_consistent:
        result["strict_exact_candidate_raw"] = "yes"
        result["strict_exact_candidate"] = "uncertain"
        repairs.append("inconsistent_strict_positive")
    if result["usable_after_relabel"] == "yes" and not (
        strict_consistent or relabel_consistent
    ):
        result["usable_after_relabel_raw"] = "yes"
        result["usable_after_relabel"] = "uncertain"
        repairs.append("inconsistent_relabel_positive")
    if enum_repairs:
        if result["strict_exact_candidate"] != "uncertain":
            result["strict_exact_candidate_raw"] = result["strict_exact_candidate"]
        if result["usable_after_relabel"] != "uncertain":
            result["usable_after_relabel_raw"] = result["usable_after_relabel"]
        result["strict_exact_candidate"] = "uncertain"
        result["usable_after_relabel"] = "uncertain"
        result["enum_repairs"] = enum_repairs
        repairs.append("unknown_enum")
    if string_repairs:
        if result["strict_exact_candidate"] != "uncertain":
            result["strict_exact_candidate_raw"] = result["strict_exact_candidate"]
        if result["usable_after_relabel"] != "uncertain":
            result["usable_after_relabel_raw"] = result["usable_after_relabel"]
        result["strict_exact_candidate"] = "uncertain"
        result["usable_after_relabel"] = "uncertain"
        result["string_repairs"] = string_repairs
        repairs.append("invalid_string_field")
    if repairs:
        result["consistency_repairs"] = sorted(set(repairs))
    return result


def successful_keys(rows: list[dict[str, Any]], model: str) -> set[str]:
    return {
        row["item_id"]
        for row in rows
        if row.get("model") == model
        and row.get("result") is not None
        and row.get("error") is None
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
    qwen_visual = qwen_record["result"]
    glm_visual = glm_record["result"]
    packet = {
        "qwen_label_blind_visual_record": qwen_visual,
        "glm_label_blind_visual_record": glm_visual,
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
        "max_tokens": 700,
        "messages": [
            {"role": "system", "content": SYSTEM_V10C},
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
            result = parse_v10c(content, qwen_visual, glm_visual)
            return {
                "item_id": metadata["item_id"],
                "uid": metadata["uid"],
                "pillar": "instructional",
                "rubric": "v10c_consensus",
                "model": model,
                "qwen_visual_model": qwen_record["model"],
                "glm_visual_model": glm_record["model"],
                "qwen_visual_record_sha256": sha256_json(qwen_visual),
                "glm_visual_record_sha256": sha256_json(glm_visual),
                "result": result,
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
        "rubric": "v10c_consensus",
        "model": model,
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--qwen-visual",
        type=Path,
        help=(
            "Optional separate Qwen V9A ledger. Required when selection rows "
            "do not embed a v9a record."
        ),
    )
    parser.add_argument("--glm-visual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8273/v1")
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    metadata_rows = load_jsonl(args.selection)
    if args.qwen_visual:
        qwen_records = latest_successful(load_jsonl(args.qwen_visual))
    else:
        qwen_records = {
            row["item_id"]: row["v9a"]
            for row in metadata_rows
            if isinstance(row.get("v9a"), dict)
            and row["v9a"].get("result") is not None
        }
    glm_records = latest_successful(load_jsonl(args.glm_visual))
    missing_qwen = [
        row["item_id"] for row in metadata_rows if row["item_id"] not in qwen_records
    ]
    if missing_qwen:
        raise ValueError(f"missing successful Qwen records: {missing_qwen[:10]}")
    missing = [row["item_id"] for row in metadata_rows if row["item_id"] not in glm_records]
    if missing:
        raise ValueError(f"missing successful GLM records: {missing[:10]}")
    previous = load_jsonl(args.out) if args.out.exists() else []
    completed = successful_keys(previous, args.model)
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
                qwen_records[row["item_id"]],
                glm_records[row["item_id"]],
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
