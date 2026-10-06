#!/usr/bin/env python3
"""Run a metadata-aware exact-violation audit over instructional storyboards."""

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


SYSTEM = """You are a fail-closed visual auditor for instructional social-norm
clips. You see a dense temporal storyboard and fallible metadata proposing a
norm with VIOLATION polarity. Decide whether that exact violation is actually
PERFORMED in the clip.

All deliberate demonstration formats are allowed: live action, role-play,
film/TV, animation, puppets, static illustrated stories, text dialogue, and
explicit etiquette contrasts. Do not demand photorealism.

Separate these questions:
1. Is there a concrete actor/character, performed action or situated utterance,
   and affected person/shared social setting in one event?
2. Is the violating action actually performed or explicitly depicted in the
   story? A narrator or on-screen text can make an illustrated action explicit,
   but merely stating "do not X", listing a rule, or describing an off-screen
   event is not performance.
3. Is the visible event itself a violation? Reject correct behavior, prevention,
   cleanup, repair, apology, advice, prohibition symbols, consequences alone,
   or an affected person's response when the proposed label says violation.
4. Does the proposed norm name the same actor's same behavior toward the same
   target? Reject labels naming the response instead of the offense.
5. Is the norm adequately concrete? Sharing, consent, personal space,
   turn-taking, sportsmanship, interrupting, apologizing, asking before
   borrowing, and littering are conventional behavior classes. Bare values
   such as respect, kindness, communication, fairness, professionalism, or
   boundaries are too broad when a more specific act is depicted.

Hard negatives include presenters/interviewees describing examples to the
audience, isolated characters singing or repeating a phrase with no recipient,
generic B-roll or ordinary tasks moralized by metadata, and formal
safety/legal/medical/technical procedures.

The storyboard is ordered left-to-right then top-to-bottom. Visible subtitles
and illustrated story text are evidence. Never infer an action solely from the
title, explanation, or proposed norm.

Return exactly one JSON object with:
performed_social_event: yes|no|uncertain
actor_action_affected_party_same_event: yes|no|uncertain
affected_party_present_in_event: yes|no|uncertain
performed_violation_not_only_described: yes|no|uncertain
proposed_actor_behavior_target_match: yes|no|uncertain
event_polarity: violation|correct|explanation_or_prohibition|neutral|uncertain
proposed_norm_relation: exact|too_broad|different_behavior|wrong_target|formal_or_non_social|uncertain
interaction_evidence: co_present_action_or_dialogue|sequential_depicted_consequence|explicit_story_event|isolated_presenter_or_character|generic_broll_or_task|uncertain
demo_format: live_action|animation|illustrated_story|text_dialogue|presenter_or_interview|broll_or_task|uncertain
exact_violation_demo: yes|no|uncertain
rejection_reason: none|no_performed_event|only_described_or_prohibited|no_affected_party|correct_or_explanatory_polarity|wrong_actor_or_target|generic_or_wrong_norm|formal_or_non_social|uncertain
literal_event: one short actor-action-target description or none
evidence: one short sentence tying ordered visual evidence to the decision

exact_violation_demo=yes only when the first five fields are yes,
event_polarity=violation, proposed_norm_relation=exact, interaction_evidence is
co_present_action_or_dialogue, sequential_depicted_consequence, or
explicit_story_event, and demo_format is live_action, animation,
illustrated_story, or text_dialogue. Otherwise fail closed."""


YNU = {"yes", "no", "uncertain"}
POLARITIES = {
    "violation",
    "correct",
    "explanation_or_prohibition",
    "neutral",
    "uncertain",
}
NORM_RELATIONS = {
    "exact",
    "too_broad",
    "different_behavior",
    "wrong_target",
    "formal_or_non_social",
    "uncertain",
}
INTERACTIONS = {
    "co_present_action_or_dialogue",
    "sequential_depicted_consequence",
    "explicit_story_event",
    "isolated_presenter_or_character",
    "generic_broll_or_task",
    "uncertain",
}
FORMATS = {
    "live_action",
    "animation",
    "illustrated_story",
    "text_dialogue",
    "presenter_or_interview",
    "broll_or_task",
    "uncertain",
}
REASONS = {
    "none",
    "no_performed_event",
    "only_described_or_prohibited",
    "no_affected_party",
    "correct_or_explanatory_polarity",
    "wrong_actor_or_target",
    "generic_or_wrong_norm",
    "formal_or_non_social",
    "uncertain",
}
QUALIFYING_INTERACTIONS = {
    "co_present_action_or_dialogue",
    "sequential_depicted_consequence",
    "explicit_story_event",
}
QUALIFYING_FORMATS = {
    "live_action",
    "animation",
    "illustrated_story",
    "text_dialogue",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "performed_social_event",
        "actor_action_affected_party_same_event",
        "affected_party_present_in_event",
        "performed_violation_not_only_described",
        "proposed_actor_behavior_target_match",
        "event_polarity",
        "proposed_norm_relation",
        "interaction_evidence",
        "demo_format",
        "exact_violation_demo",
        "rejection_reason",
        "literal_event",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    atomic = [
        "performed_social_event",
        "actor_action_affected_party_same_event",
        "affected_party_present_in_event",
        "performed_violation_not_only_described",
        "proposed_actor_behavior_target_match",
        "exact_violation_demo",
    ]
    for key in atomic:
        if result[key] not in YNU:
            raise ValueError(f"invalid {key}: {result[key]!r}")
    enum_specs = {
        "event_polarity": POLARITIES,
        "proposed_norm_relation": NORM_RELATIONS,
        "interaction_evidence": INTERACTIONS,
        "demo_format": FORMATS,
        "rejection_reason": REASONS,
    }
    repairs: list[str] = []
    for key, allowed in enum_specs.items():
        if result[key] not in allowed:
            result[f"{key}_raw"] = result[key]
            result[key] = "uncertain"
            repairs.append(key)
    string_repairs: list[str] = []
    for key in ("literal_event", "evidence"):
        if not isinstance(result[key], str) or not result[key].strip():
            result[f"{key}_raw"] = result[key]
            result[key] = "none"
            string_repairs.append(key)

    strict = (
        all(result[key] == "yes" for key in atomic[:5])
        and result["event_polarity"] == "violation"
        and result["proposed_norm_relation"] == "exact"
        and result["interaction_evidence"] in QUALIFYING_INTERACTIONS
        and result["demo_format"] in QUALIFYING_FORMATS
        and result["rejection_reason"] == "none"
    )
    if result["exact_violation_demo"] == "yes" and not strict:
        result["exact_violation_demo_raw"] = "yes"
        result["exact_violation_demo"] = "uncertain"
        repairs.append("inconsistent_positive")
    if string_repairs and result["exact_violation_demo"] == "yes":
        result["exact_violation_demo_raw"] = "yes"
        result["exact_violation_demo"] = "uncertain"
        repairs.append("invalid_string_in_positive")
    if string_repairs:
        result["string_repairs"] = string_repairs
    if repairs:
        result["validation_repairs"] = sorted(set(repairs))
        result["exact_violation_demo"] = "uncertain"
    return result


def successful(rows: list[dict[str, Any]], model: str) -> set[str]:
    return {
        row["item_id"]
        for row in rows
        if row.get("model") == model
        and row.get("error") is None
        and row.get("result") is not None
    }


def request_one(
    endpoint: str,
    model: str,
    row: dict[str, Any],
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    sheet = Path(row["sheet_path"])
    if not sheet.is_absolute():
        sheet = Path(row["_manifest_dir"]) / sheet
    if file_sha256(sheet) != row["sheet_sha256"]:
        raise ValueError(f"storyboard hash mismatch: {row['item_id']}")
    metadata = {
        "title": row.get("title"),
        "category": row.get("category"),
        "proposed_norm": row.get("norm"),
        "proposed_polarity": row.get("polarity"),
        "start_quote": row.get("start_quote"),
        "end_quote": row.get("end_quote"),
        "explanation": row.get("explanation"),
    }
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 700,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": sheet.resolve().as_uri()},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Audit the ordered storyboard against this fallible "
                            "metadata. Return only the required JSON.\n"
                            + json.dumps(
                                metadata, ensure_ascii=False, sort_keys=True
                            )
                        ),
                    },
                ],
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
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": "instructional",
                "rubric": "v11_metadata_vlm",
                "model": model,
                "sheet_sha256": row["sheet_sha256"],
                "result": parse_result(content),
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
        "pillar": "instructional",
        "rubric": "v11_metadata_vlm",
        "model": model,
        "sheet_sha256": row["sheet_sha256"],
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyboards", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    storyboard_rows = load_jsonl(args.storyboards)
    metadata = {row["item_id"]: row for row in load_jsonl(args.metadata)}
    rows: list[dict[str, Any]] = []
    for storyboard in storyboard_rows:
        item_id = storyboard["item_id"]
        if item_id not in metadata:
            raise ValueError(f"missing metadata for {item_id}")
        merged = {**metadata[item_id], **storyboard}
        merged["_manifest_dir"] = str(args.storyboards.parent)
        rows.append(merged)
    previous = load_jsonl(args.out) if args.out.exists() else []
    completed = successful(previous, args.model)
    pending = [row for row in rows if row["item_id"] not in completed]
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
