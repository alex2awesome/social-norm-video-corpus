#!/usr/bin/env python3
"""Visual-only reranking of localized witnessed reaction candidates (V2)."""

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


SYSTEM = """You audit visual evidence around one transcript-localized candidate
moment. The storyboard is SILENT: never infer who spoke from the supplied
candidate text, never claim that something is audible, and never use the text
to assign a person's role. Frames are chronological left-to-right then
top-to-bottom, with timestamps.

Look only for a visually depicted triggering action and a visibly distinct
third person's response: approaching/interposing, stopping or separating
people, pointed objection gestures, protective aid, summoning help, or an
official sanction. A bystander is neither the actor nor the directly affected
target. Do not count the actor's defense, the target's complaint/distress,
camera motion, narration, generic laughter/surprise/fear, or concern after an
accident. If the responder cannot be visually identified, answer uncertain or
no; the transcript is not identity evidence.

Return exactly one JSON object:
triggering_action_visible: yes|no|uncertain
visible_response_at_candidate_moment: yes|no|uncertain
response_after_or_overlaps_action: yes|no|uncertain
visibly_distinct_third_person_responds: yes|no|uncertain
visible_response_targets_action: yes|no|uncertain
visual_responder_role: bystander|organic_audience|authority_or_host|affected_target|violator_or_actor|uncertain|none
visible_response_content: targeted_objection|correction_or_sanction|protective_intervention|interposition_or_separation|generic_affect|self_defense_or_excuse|description_only|none|uncertain
strict_visual_bystander_response: yes|no|uncertain
evidence: one short sentence naming only visible evidence

strict_visual_bystander_response=yes only when the first five atoms are yes,
role is bystander or organic_audience, and content is targeted objection,
correction/sanction, protective intervention, or interposition/separation.
Authorities are reported separately. Fail closed."""

TRINARY = {"yes", "no", "uncertain"}
ROLES = {
    "bystander",
    "organic_audience",
    "authority_or_host",
    "affected_target",
    "violator_or_actor",
    "uncertain",
    "none",
}
CONTENTS = {
    "targeted_objection",
    "correction_or_sanction",
    "protective_intervention",
    "interposition_or_separation",
    "generic_affect",
    "self_defense_or_excuse",
    "description_only",
    "none",
    "uncertain",
}
ATOMS = (
    "triggering_action_visible",
    "visible_response_at_candidate_moment",
    "response_after_or_overlaps_action",
    "visibly_distinct_third_person_responds",
    "visible_response_targets_action",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_result(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        result = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        result = json.loads(stripped[start : end + 1])
    strict_key = "strict_visual_bystander_response"
    expected = set(ATOMS) | {
        "visual_responder_role",
        "visible_response_content",
        strict_key,
        "evidence",
    }
    if not isinstance(result, dict) or set(result) != expected:
        raise ValueError("visual candidate VLM schema mismatch")
    for key in (*ATOMS, strict_key):
        if result[key] not in TRINARY:
            raise ValueError(f"invalid {key}: {result[key]!r}")
    if result["visual_responder_role"] not in ROLES:
        raise ValueError("invalid visual_responder_role")
    if result["visible_response_content"] not in CONTENTS:
        raise ValueError("invalid visible_response_content")
    if not isinstance(result["evidence"], str) or not result["evidence"].strip():
        raise ValueError("evidence must be non-empty")
    consistent = (
        all(result[key] == "yes" for key in ATOMS)
        and result["visual_responder_role"] in {"bystander", "organic_audience"}
        and result["visible_response_content"]
        in {
            "targeted_objection",
            "correction_or_sanction",
            "protective_intervention",
            "interposition_or_separation",
        }
    )
    if result[strict_key] == "yes" and not consistent:
        result[f"{strict_key}_raw"] = "yes"
        result[strict_key] = "uncertain"
        result["consistency_repair"] = "inconsistent_positive_to_uncertain"
    return result


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
    if sha256(sheet) != row["sheet_sha256"]:
        raise ValueError(f"storyboard hash mismatch: {row['candidate_id']}")
    prompt = (
        f"The transcript candidate is localized at "
        f"{row['candidate_start_sec']:.2f}-{row['candidate_end_sec']:.2f}s. "
        "Its text is supplied only as a timing identifier and is not speaker "
        f"or role evidence: {json.dumps(row['candidate_text'], ensure_ascii=False)}. "
        "Audit visual behavior at that moment from the silent storyboard."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 450,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": sheet.resolve().as_uri()},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode()
    last_error, last_content = "", None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            last_content = body["choices"][0]["message"].get("content")
            if not last_content:
                raise ValueError("empty response")
            return {
                "candidate_id": row["candidate_id"],
                "item_id": row["item_id"],
                "uid": row["uid"],
                "model": model,
                "rubric": "witnessed_reaction_candidate_visual_v2",
                "sheet_sha256": row["sheet_sha256"],
                "candidate_text": row["candidate_text"],
                "result": parse_result(last_content),
                "raw_response": last_content,
                "usage": body.get("usage"),
                "error": None,
            }
        except (
            urllib.error.HTTPError,
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
        "candidate_id": row["candidate_id"],
        "item_id": row["item_id"],
        "uid": row["uid"],
        "model": model,
        "rubric": "witnessed_reaction_candidate_visual_v2",
        "sheet_sha256": row["sheet_sha256"],
        "candidate_text": row["candidate_text"],
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8271/v1")
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    rows = load_jsonl(args.manifest)
    for row in rows:
        row["_manifest_dir"] = str(args.manifest.parent)
    completed = {
        (row["candidate_id"], row["model"])
        for row in load_jsonl(args.out)
        if row.get("error") is None
    } if args.out.exists() else set()
    pending = [
        row for row in rows
        if (row["candidate_id"], args.model) not in completed
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
                args.timeout,
                args.retries,
            ): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{index}/{len(pending)} {result['candidate_id']} "
                f"{'ok' if result['error'] is None else result['error']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
