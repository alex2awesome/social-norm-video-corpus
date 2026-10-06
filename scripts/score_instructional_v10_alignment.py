#!/usr/bin/env python3
"""Run the text/alignment pass for instructional V10 shadow discovery."""

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


SYSTEM_V10B = """You are the second, TEXT-ONLY alignment pass of a video audit.
You receive:
1. a frozen label-blind visual record from another model; and
2. fallible metadata containing a proposed norm, polarity, transcript quotes,
   explanation, title, and search category.

Do not invent visual evidence. The visual record is the entire visual evidence
boundary. It may itself be wrong, so read its literal actor, action, affected
party, scope, scene role, and evidence critically.

The project wants a concrete performed demonstration of tacit interpersonal
conduct, shared-public conduct, or explicit social etiquette. Any deliberate
format is acceptable. Formal safety/traffic/legal rules, medical or service
procedures, developmental exercises, diagnostic tasks, and ordinary object
actions are not social-norm demos unless a distinct socially evaluable behavior
is performed.

Judge label specificity strictly:
- `exact_concrete` names the performed conduct and correct affected party;
- `broader_but_informative` is true but loses the concrete conduct;
- `generic_vacuous` includes labels such as bare "respect", "customer
  service", "professionalism", "communication", "behavior", "care", or
  "boundaries" when the actual act is a specific insult, bias, coercion,
  dishonesty, refusal, or abuse;
- `different_behavior` names another act;
- `wrong_affected_party` names the wrong recipient or role;
- `formal_or_non_social` moralizes a safety, procedure, skill, or task.

Examples of alignment distinctions, not facts about the current item:
- an insult to a server is not exactly labeled by bare "respect", but can be
  safely relabeled "do not abuse service workers";
- a supervisor berating interns is not exactly "respect patients";
- a dangerous road crossing is not tacit conduct;
- a presenter saying a sample rude phrase to the audience is not a situated
  demo;
- a performed bow or handshake can be explicit social etiquette.

`exact_original_usable=yes` requires all of:
- frozen visual `demo_usable=yes`;
- a qualifying social scope and performed behavior;
- exact_concrete norm specificity;
- actor/action/affected-party alignment=yes; and
- proposed polarity matches the performed event.

`usable_after_relabel=yes` still requires a qualifying performed demo, but may
repair broader/generic/different labels only when `safe_relabel` names the
literal event. It may never rescue description-only, procedural, developmental,
or non-social imagery.

Return one JSON object with exactly:
visual_record_supports_demo: yes|no|uncertain
qualifying_social_scope: yes|no|uncertain
target_behavior_performed_not_merely_described: yes|no|uncertain
actor_action_affected_party_alignment: yes|no|uncertain
proposed_norm_specificity: exact_concrete|broader_but_informative|generic_vacuous|different_behavior|wrong_affected_party|formal_or_non_social|uncertain
proposed_polarity_relation: matches_event|opposes_event|explanation_not_event_polarity|uncertain
visible_event_polarity: violation|correct|explanation|neutral_or_ambiguous|uncertain
exact_original_usable: yes|no|uncertain
usable_after_relabel: yes|no|uncertain
safe_relabel: short concrete norm label or none
failure_mechanism: none|visual_gate_failed|formal_or_non_social|behavior_only_described|generic_label|different_behavior|wrong_affected_party|polarity_mismatch|uncertain
alignment_evidence: one short sentence comparing the literal visual event with the proposed label

The fields must agree. Fail closed on every uncertainty."""


YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}
SPECIFICITIES = {
    "exact_concrete",
    "broader_but_informative",
    "generic_vacuous",
    "different_behavior",
    "wrong_affected_party",
    "formal_or_non_social",
    "uncertain",
}
POLARITY_RELATIONS = {
    "matches_event",
    "opposes_event",
    "explanation_not_event_polarity",
    "uncertain",
}
VISIBLE_POLARITIES = {
    "violation",
    "correct",
    "explanation",
    "neutral_or_ambiguous",
    "uncertain",
}
FAILURES = {
    "none",
    "visual_gate_failed",
    "formal_or_non_social",
    "behavior_only_described",
    "generic_label",
    "different_behavior",
    "wrong_affected_party",
    "polarity_mismatch",
    "uncertain",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


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


def parse_v10b(text: str, visual_result: dict[str, Any]) -> dict[str, Any]:
    result = _json_object(text)
    expected = {
        "visual_record_supports_demo",
        "qualifying_social_scope",
        "target_behavior_performed_not_merely_described",
        "actor_action_affected_party_alignment",
        "proposed_norm_specificity",
        "proposed_polarity_relation",
        "visible_event_polarity",
        "exact_original_usable",
        "usable_after_relabel",
        "safe_relabel",
        "failure_mechanism",
        "alignment_evidence",
    }
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    atomic = [
        "visual_record_supports_demo",
        "qualifying_social_scope",
        "target_behavior_performed_not_merely_described",
        "actor_action_affected_party_alignment",
        "exact_original_usable",
        "usable_after_relabel",
    ]
    for key in atomic:
        if result[key] not in YES_NO_UNCERTAIN:
            raise ValueError(f"invalid {key}: {result[key]!r}")
    enum_specs = {
        "proposed_norm_specificity": (SPECIFICITIES, "uncertain"),
        "proposed_polarity_relation": (POLARITY_RELATIONS, "uncertain"),
        "visible_event_polarity": (VISIBLE_POLARITIES, "uncertain"),
        "failure_mechanism": (FAILURES, "uncertain"),
    }
    repairs: list[str] = []
    for key, (allowed, fallback) in enum_specs.items():
        if result[key] not in allowed:
            result[f"{key}_raw"] = result[key]
            result[key] = fallback
            repairs.append(key)
    if not isinstance(result["safe_relabel"], str) or not result["safe_relabel"].strip():
        raise ValueError("safe_relabel must be a non-empty string or 'none'")
    if not isinstance(result["alignment_evidence"], str) or not result[
        "alignment_evidence"
    ].strip():
        raise ValueError("alignment_evidence must be non-empty")

    visual_pass = visual_result.get("demo_usable") == "yes"
    exact_consistent = (
        visual_pass
        and result["visual_record_supports_demo"] == "yes"
        and result["qualifying_social_scope"] == "yes"
        and result["target_behavior_performed_not_merely_described"] == "yes"
        and result["actor_action_affected_party_alignment"] == "yes"
        and result["proposed_norm_specificity"] == "exact_concrete"
        and result["proposed_polarity_relation"] == "matches_event"
    )
    relabel_consistent = (
        visual_pass
        and result["visual_record_supports_demo"] == "yes"
        and result["qualifying_social_scope"] == "yes"
        and result["target_behavior_performed_not_merely_described"] == "yes"
        and result["usable_after_relabel"] == "yes"
        and result["safe_relabel"].lower() != "none"
    )
    consistency_repairs: list[str] = []
    if result["exact_original_usable"] == "yes" and not exact_consistent:
        result["exact_original_usable_raw"] = "yes"
        result["exact_original_usable"] = "uncertain"
        consistency_repairs.append("inconsistent_exact_positive")
    if result["usable_after_relabel"] == "yes" and not (
        exact_consistent or relabel_consistent
    ):
        result["usable_after_relabel_raw"] = "yes"
        result["usable_after_relabel"] = "uncertain"
        consistency_repairs.append("inconsistent_relabel_positive")
    if not visual_pass:
        if result["exact_original_usable"] == "yes":
            result["exact_original_usable_raw"] = "yes"
        if result["usable_after_relabel"] == "yes":
            result["usable_after_relabel_raw"] = "yes"
        result["exact_original_usable"] = "uncertain"
        result["usable_after_relabel"] = "uncertain"
        consistency_repairs.append("visual_gate_not_positive")
    if repairs:
        if result["exact_original_usable"] != "uncertain":
            result["exact_original_usable_raw"] = result["exact_original_usable"]
        if result["usable_after_relabel"] != "uncertain":
            result["usable_after_relabel_raw"] = result["usable_after_relabel"]
        result["exact_original_usable"] = "uncertain"
        result["usable_after_relabel"] = "uncertain"
        result["enum_repairs"] = repairs
        consistency_repairs.append("unknown_enum")
    if consistency_repairs:
        result["consistency_repairs"] = sorted(set(consistency_repairs))
    return result


def latest_successful(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("error") is None and row.get("result") is not None:
            latest[row["item_id"]] = row
    return latest


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
    visual: dict[str, Any],
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    visual_result = visual["result"]
    packet = {
        "frozen_label_blind_visual_record": visual_result,
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
        "max_tokens": 750,
        "messages": [
            {"role": "system", "content": SYSTEM_V10B},
            {
                "role": "user",
                "content": (
                    "Audit this frozen packet. Return only the required JSON.\n"
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
            result = parse_v10b(content, visual_result)
            return {
                "item_id": metadata["item_id"],
                "uid": metadata["uid"],
                "pillar": "instructional",
                "rubric": "v10b_alignment",
                "model": model,
                "visual_model": visual["model"],
                "visual_record_sha256": sha256_json(visual_result),
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
        "rubric": "v10b_alignment",
        "model": model,
        "visual_model": visual["model"],
        "visual_record_sha256": sha256_json(visual_result),
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8273/v1")
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    metadata_rows = load_jsonl(args.selection)
    visuals = latest_successful(load_jsonl(args.visual))
    missing = [row["item_id"] for row in metadata_rows if row["item_id"] not in visuals]
    if missing:
        raise ValueError(f"missing successful visual records: {missing[:10]}")
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
                visuals[row["item_id"]],
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
