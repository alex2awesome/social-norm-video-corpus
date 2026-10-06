#!/usr/bin/env python3
"""Role-aware VLM reranking for localized witnessed reaction candidates."""

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


SYSTEM = """You audit one localized candidate reaction in a video storyboard.
Frames are chronological left-to-right then top-to-bottom. The candidate
utterance and its timestamp are supplied only to localize the possible
reaction; they do not prove who said it or that it responds to an action.

Identify literal roles. A bystander is a third person who is neither the actor
performing the triggering behavior nor its directly affected target. An
authority_or_host acts in an official venue role. Do not relabel the affected
target's complaint, the actor's excuse/apology, narration, dialogue quoted by
a presenter, laughter, surprise, fear, or generic concern as a bystander
reaction. The triggering action may be scripted, but it must occur in the same
depicted episode before or overlapping the response.

Return exactly one JSON object:
candidate_grounded: yes|no|uncertain
triggering_action_visible_or_audible: yes|no|uncertain
action_before_or_overlaps_candidate: yes|no|uncertain
reaction_targets_action: yes|no|uncertain
reaction_source_role: bystander|organic_audience|authority_or_host|affected_target|violator_or_actor|narrator_or_commentator|uncertain|none
reaction_content: targeted_objection|correction_or_sanction|protective_intervention|generic_affect|self_defense_or_excuse|compliance_or_apology|description_only|none|uncertain
reactor_visibly_distinct_from_actor: yes|no|uncertain
strict_bystander_reaction: yes|no|uncertain
evidence: one short literal sentence

strict_bystander_reaction=yes only if the first four atoms and visibly-distinct
atom are yes, role is bystander or organic_audience, and content is targeted
objection, correction/sanction, or protective intervention. Authorities are
reported separately and are not strict bystanders. Fail closed."""

TRINARY = {"yes", "no", "uncertain"}
ROLES = {
    "bystander",
    "organic_audience",
    "authority_or_host",
    "affected_target",
    "violator_or_actor",
    "narrator_or_commentator",
    "uncertain",
    "none",
}
CONTENTS = {
    "targeted_objection",
    "correction_or_sanction",
    "protective_intervention",
    "generic_affect",
    "self_defense_or_excuse",
    "compliance_or_apology",
    "description_only",
    "none",
    "uncertain",
}
ATOMS = (
    "candidate_grounded",
    "triggering_action_visible_or_audible",
    "action_before_or_overlaps_candidate",
    "reaction_targets_action",
    "reactor_visibly_distinct_from_actor",
    "strict_bystander_reaction",
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
    expected = set(ATOMS) | {"reaction_source_role", "reaction_content", "evidence"}
    if not isinstance(result, dict) or set(result) != expected:
        raise ValueError("candidate VLM schema mismatch")
    for key in ATOMS:
        if result[key] not in TRINARY:
            raise ValueError(f"invalid {key}: {result[key]!r}")
    if result["reaction_source_role"] not in ROLES:
        raise ValueError("invalid reaction_source_role")
    if result["reaction_content"] not in CONTENTS:
        raise ValueError("invalid reaction_content")
    if not isinstance(result["evidence"], str) or not result["evidence"].strip():
        raise ValueError("evidence must be non-empty")
    strict_atoms = all(result[key] == "yes" for key in ATOMS[:-1])
    strict_role = result["reaction_source_role"] in {
        "bystander",
        "organic_audience",
    }
    strict_content = result["reaction_content"] in {
        "targeted_objection",
        "correction_or_sanction",
        "protective_intervention",
    }
    if result["strict_bystander_reaction"] == "yes" and not (
        strict_atoms and strict_role and strict_content
    ):
        result["strict_bystander_reaction_raw"] = "yes"
        result["strict_bystander_reaction"] = "uncertain"
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
        f"Candidate utterance at {row['candidate_start_sec']:.2f}-"
        f"{row['candidate_end_sec']:.2f}s: "
        f"{json.dumps(row['candidate_text'], ensure_ascii=False)}. "
        "Audit this candidate using the complete temporal storyboard."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 500,
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
                "rubric": "witnessed_reaction_candidate_v1",
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
        "rubric": "witnessed_reaction_candidate_v1",
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
