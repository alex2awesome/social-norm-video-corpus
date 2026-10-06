#!/usr/bin/env python3
"""Run an append-only local Ollama audit over frozen temporal contact sheets."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCHEMA = {
    "type": "object",
    "properties": {
        "social_behavior_scene_visible": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
        "target_pillar_signal_visible": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
        "observable_action_visible": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
        "interaction_or_shared_context_visible": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
        "action_then_causal_response_visible": {
            "type": "string",
            "enum": ["yes", "no", "uncertain", "na"],
        },
        "commentary_event_visual_pair_present": {
            "type": "string",
            "enum": ["yes", "no", "uncertain", "na"],
        },
        "presentation_or_context_only": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
        "hypothesized_behavior_directly_visible": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
        "behavior_intentionally_demonstrated_or_exemplified": {
            "type": "string",
            "enum": ["yes", "no", "uncertain", "na"],
        },
        "medium": {
            "type": "string",
            "enum": [
                "live_action",
                "animation",
                "illustrated_story",
                "screen_scenario",
                "talking_head",
                "broll",
                "graphics",
                "mixed",
                "uncertain",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
    "required": [
        "social_behavior_scene_visible",
        "target_pillar_signal_visible",
        "observable_action_visible",
        "interaction_or_shared_context_visible",
        "action_then_causal_response_visible",
        "commentary_event_visual_pair_present",
        "presentation_or_context_only",
        "hypothesized_behavior_directly_visible",
        "behavior_intentionally_demonstrated_or_exemplified",
        "medium",
        "confidence",
        "evidence",
    ],
}

SYSTEM = """You audit temporal contact sheets from videos. Frames are ordered
left-to-right and then top-to-bottom; timestamps are printed in each cell.
Use only visible pixels across the sampled times. A title, query, caption, or
prompted label is a fallible hypothesis, never proof.

A social behavior scene visibly shows a concrete socially meaningful action,
interaction, or situated speech context. It may be live action, staged,
animated, illustrated, or screen-mediated. A presenter, interview, generic
B-roll, aftermath, setting, text card, or unrelated person is not itself a
behavior scene.

Instructional accepts any medium only when it demonstrates the behavior.
Witnessed requires the alleged action followed by a causal contemporaneous
reaction; action-only footage is not the full witnessed signal.
Commentary requires both an event depiction and visible commentary/presentation
context in the source. A contact sheet cannot prove spoken semantic alignment,
so use uncertain when that is the missing evidence.

Use uncertain only when visible evidence is genuinely ambiguous, not as a
default. Apply these categorical invariants:
- If the sheet is only text/graphics, generic B-roll, interviews, or a presenter
  with no depicted behavior, social_behavior_scene_visible=no.
- If no action is visible, observable_action_visible=no.
- If no interaction/shared action context is visible,
  interaction_or_shared_context_visible=no.
- For a non-witnessed candidate, action_then_causal_response_visible=na.
- For a non-commentary candidate, commentary_event_visual_pair_present=na.
- If the sheet only presents/explains context, presentation_or_context_only=yes.
- target_pillar_signal_visible=uncertain in blind mode only because no target
  hypothesis is supplied.
- In conditioned instructional mode, target_pillar_signal_visible=yes when a
  visible scene demonstrates the hypothesized behavior. It does not need an
  explicit "training" logo, instructor, or lesson card. Set it to yes only when
  BOTH hypothesized_behavior_directly_visible=yes AND
  behavior_intentionally_demonstrated_or_exemplified=yes.
- In conditioned witnessed mode, target_pillar_signal_visible=yes only when the
  hypothesized action and a causal contemporaneous response are both visible.
- In conditioned commentary mode, target_pillar_signal_visible=yes only when
  event imagery and commentary/presentation context are both visible; use
  uncertain if spoken semantic alignment is the sole missing fact.
- In any conditioned mode, if your evidence says no target scene is depicted,
  target_pillar_signal_visible=no. If your evidence says the visible scene
  supports the hypothesis, target_pillar_signal_visible=yes. Do not return
  uncertain merely because the source lacks explicit pillar branding.

For instructional candidates, separate two common confusions:
- hypothesized_behavior_directly_visible asks whether the specific proposed
  behavior itself is shown, not merely named in captions, discussed, implied by
  a setting, or occurring as generic conversation.
- behavior_intentionally_demonstrated_or_exemplified asks whether the source
  presents the behavior as an example/story/scenario. Role-play, scripted
  scenes, animation, puppets, illustrated stories, and an organic example
  explicitly used as a demonstration qualify. An ordinary panel discussion
  does not become a turn-taking demo merely because panelists take turns.
  A presenter listing anti-bullying tips does not demonstrate intervention
  unless the intervention is also depicted.
For non-instructional candidates set
behavior_intentionally_demonstrated_or_exemplified=na.

Be conservative about brief events that the sparse sheet may miss, but do not
override clear negative evidence with blanket uncertainty. Return only the
requested JSON."""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def successful_prediction_keys(records: list[dict]) -> set[tuple[str, str, str, str]]:
    return {
        (
            row["item_id"],
            row["mode"],
            row["model"],
            row.get("rubric_version", "contact_sheet_v1"),
        )
        for row in records
        if row.get("error") is None and row.get("result") is not None
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_for(row: dict, mode: str) -> str:
    if mode == "blind":
        return (
            f"This is a candidate for the {row['pillar']} collection. No target "
            "behavior is supplied. Judge whether any concrete social behavior "
            "scene is visible. Set target_pillar_signal_visible to uncertain."
        )
    return (
        f"Candidate pillar: {row['pillar']}.\n"
        f"Retrieval hypothesis: {row['query']!r}.\n"
        "Judge whether the visible sequence supports the pillar-specific signal "
        "and retrieval hypothesis. Do not accept based on these words."
    )


def validate_result(result: dict, row: dict, mode: str) -> None:
    missing = set(SCHEMA["required"]) - result.keys()
    if missing:
        raise ValueError(f"response missing keys: {sorted(missing)}")
    for key, definition in SCHEMA["properties"].items():
        if "enum" in definition and result[key] not in definition["enum"]:
            raise ValueError(f"invalid {key}: {result[key]!r}")
    confidence = float(result["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError(f"confidence out of range: {confidence}")
    if mode == "blind" and result["target_pillar_signal_visible"] != "uncertain":
        raise ValueError("blind target_pillar_signal_visible must be uncertain")
    if (
        row["pillar"] != "witnessed"
        and result["action_then_causal_response_visible"] != "na"
    ):
        raise ValueError("non-witnessed action/response field must be na")
    if (
        row["pillar"] != "commentary"
        and result["commentary_event_visual_pair_present"] != "na"
    ):
        raise ValueError("non-commentary event-pair field must be na")
    if (
        row["pillar"] != "instructional"
        and result["behavior_intentionally_demonstrated_or_exemplified"] != "na"
    ):
        raise ValueError("non-instructional demonstration field must be na")
    if (
        mode == "conditioned"
        and row["pillar"] == "instructional"
        and result["target_pillar_signal_visible"] == "yes"
        and (
            result["hypothesized_behavior_directly_visible"] != "yes"
            or result["behavior_intentionally_demonstrated_or_exemplified"] != "yes"
        )
    ):
        raise ValueError("instructional yes requires visible behavior and demonstration")


def request_one(
    endpoint: str,
    model: str,
    row: dict,
    mode: str,
    rubric_version: str,
    timeout: int,
    retries: int,
) -> dict:
    sheet = Path(row["sheet_path"])
    if sha256(sheet) != row["sheet_sha256"]:
        raise ValueError(f"sheet hash mismatch: {row['item_id']}")
    image = base64.b64encode(sheet.read_bytes()).decode()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": prompt_for(row, mode),
                "images": [image],
            },
        ],
        "format": SCHEMA,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 7},
        "keep_alive": "30m",
    }
    encoded = json.dumps(payload).encode()
    last_error = ""
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/api/chat",
            data=encoded,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            result = json.loads(body["message"]["content"])
            validate_result(result, row, mode)
            return {
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": row["pillar"],
                "mode": mode,
                "model": model,
                "rubric_version": rubric_version,
                "gold_visual_content_present": row["gold_visual_content_present"],
                "gold_social_behavior_scene": row["gold_social_behavior_scene"],
                "gold_target_source_pass": row["gold_target_source_pass"],
                "result": result,
                "timing": {
                    key: body.get(key)
                    for key in (
                        "total_duration",
                        "load_duration",
                        "prompt_eval_count",
                        "prompt_eval_duration",
                        "eval_count",
                        "eval_duration",
                    )
                },
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
        "item_id": row["item_id"],
        "uid": row["uid"],
        "pillar": row["pillar"],
        "mode": mode,
        "model": model,
        "rubric_version": rubric_version,
        "gold_visual_content_present": row["gold_visual_content_present"],
        "gold_social_behavior_scene": row["gold_social_behavior_scene"],
        "gold_target_source_pass": row["gold_target_source_pass"],
        "result": None,
        "timing": None,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="ollama.com/library/qwen3-vl:8b-instruct")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--mode", choices=("blind", "conditioned"), required=True)
    parser.add_argument("--rubric-version", default="contact_sheet_v4")
    parser.add_argument("--pillar", choices=("instructional", "witnessed", "commentary"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    completed = (
        successful_prediction_keys(load_jsonl(args.out))
        if args.out.exists()
        else set()
    )
    pending = [
        row
        for row in rows
        if (row["item_id"], args.mode, args.model, args.rubric_version)
        not in completed
        and (args.pillar is None or row["pillar"] == args.pillar)
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.out.open("a") as handle,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {
            pool.submit(
                request_one,
                args.endpoint,
                args.model,
                row,
                args.mode,
                args.rubric_version,
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


if __name__ == "__main__":
    main()
