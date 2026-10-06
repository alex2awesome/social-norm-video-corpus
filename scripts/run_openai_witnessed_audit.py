#!/usr/bin/env python3
"""Audit witnessed action→reaction candidates with structured GPT vision output."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


PROMPT_VERSION = "witnessed_v1"
SYSTEM_PROMPT = """You audit candidate witnessed social-norm violations.
The chronological frames and aligned transcript cover a clip ending near a
detector-selected reaction quote. The reaction is weak supervision, not the
training input. Determine whether an observable social action occurs first and
causes an on-scene person's normative reaction.

Accept witnessed supervision only for organic or genuinely hidden-camera events.
Movie/TV scenes, skits, animation, staged morality stories, reenactments, news,
streamer commentary, and narration are not witnessed positives. A scripted scene
that clearly demonstrates the labeled behavior may be marked reroute_instructional
instead. Generic surprise, excitement, profanity, pain, fear, or confusion is not
a normative reaction unless it targets concrete social conduct.

Do not infer an action from the query, title, reaction quote, or narration when it
is not visible. A usable candidate must contain a complete pre-reaction action that
remains intelligible after the reaction is removed. Estimate clip-relative seconds
for the end of the action and start of the reaction. Use uncertain when frames do
not establish causality, authenticity, or clean splice bounds."""


TRI = ["yes", "no", "uncertain"]
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_social_norm": {"type": "string", "enum": TRI},
        "social_action_visible": {"type": "string", "enum": TRI},
        "reaction_visible": {"type": "string", "enum": TRI},
        "action_before_reaction": {"type": "string", "enum": TRI},
        "reaction_targets_action": {"type": "string", "enum": TRI},
        "reaction_is_normative": {"type": "string", "enum": TRI},
        "behavior_label_supported": {"type": "string", "enum": TRI},
        "clean_pre_reaction_demo": {"type": "string", "enum": TRI},
        "authenticity": {
            "type": "string",
            "enum": [
                "organic",
                "hidden_camera_genuine",
                "scripted",
                "animation",
                "news_or_commentary",
                "uncertain",
            ],
        },
        "decision": {
            "type": "string",
            "enum": ["accept_after_splice", "reroute_instructional", "reject", "uncertain"],
        },
        "action_end_sec": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "reaction_start_sec": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "rejection_reasons": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "no_social_action",
                    "no_reaction",
                    "reaction_not_normative",
                    "reaction_not_targeted",
                    "causality_not_visible",
                    "label_mismatch",
                    "scripted_or_staged",
                    "news_or_commentary",
                    "generic_affect",
                    "no_clean_pre_reaction_clip",
                    "insufficient_temporal_evidence",
                    "off_topic",
                    "other",
                ],
            },
        },
        "action_evidence_frames": {"type": "array", "items": {"type": "integer"}},
        "reaction_evidence_frames": {"type": "array", "items": {"type": "integer"}},
        "description": {"type": "string"},
    },
    "required": [
        "is_social_norm",
        "social_action_visible",
        "reaction_visible",
        "action_before_reaction",
        "reaction_targets_action",
        "reaction_is_normative",
        "behavior_label_supported",
        "clean_pre_reaction_demo",
        "authenticity",
        "decision",
        "action_end_sec",
        "reaction_start_sec",
        "rejection_reasons",
        "action_evidence_frames",
        "reaction_evidence_frames",
        "description",
    ],
    "additionalProperties": False,
}


def prompt_hash() -> str:
    value = {"version": PROMPT_VERSION, "system": SYSTEM_PROMPT, "schema": SCHEMA}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def content_for_item(root: Path, item: dict[str, Any]) -> list[dict[str, Any]]:
    context = {
        "title": item.get("title"),
        "category": item.get("category"),
        "found_by_query": item.get("found_by_query"),
        "detector_tag": item.get("polarity"),
        "proposed_norm": item.get("norm"),
        "reaction_quote": item.get("start_quote"),
        "reaction_context": item.get("explanation"),
        "duration_seconds": item.get("duration"),
        "aligned_transcript": item.get("aligned_transcript") or [],
        "frames": [
            {"frame_index": frame["frame_index"], "timestamp": frame["timestamp"]}
            for frame in item["frames"]
        ],
    }
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": "Candidate context:\n" + json.dumps(context, ensure_ascii=False)}
    ]
    for frame in item["frames"]:
        content.extend(
            [
                {"type": "input_text", "text": f"Frame {frame['frame_index']} at {frame['timestamp']:.3f}s:"},
                {"type": "input_image", "image_url": data_url(root / frame["path"]), "detail": "high"},
            ]
        )
    return content


def validate_result(result: dict[str, Any], item: dict[str, Any]) -> None:
    missing = set(SCHEMA["required"]) - set(result)
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    valid_frames = {frame["frame_index"] for frame in item["frames"]}
    evidence = set(result["action_evidence_frames"]) | set(result["reaction_evidence_frames"])
    if not evidence.issubset(valid_frames):
        raise ValueError("evidence contains an invalid frame index")
    if result["decision"] == "accept_after_splice":
        required_yes = (
            "is_social_norm",
            "social_action_visible",
            "reaction_visible",
            "action_before_reaction",
            "reaction_targets_action",
            "reaction_is_normative",
            "behavior_label_supported",
            "clean_pre_reaction_demo",
        )
        if any(result[key] != "yes" for key in required_yes):
            raise ValueError("accepted witnessed item violates yes/no invariants")
        if result["authenticity"] not in {"organic", "hidden_camera_genuine"}:
            raise ValueError("accepted witnessed item is not organic")
        action_end, reaction_start = result["action_end_sec"], result["reaction_start_sec"]
        if action_end is None or reaction_start is None:
            raise ValueError("accepted witnessed item requires splice bounds")
        duration = float(item["duration"])
        if not 0 < float(action_end) <= float(reaction_start) < duration:
            raise ValueError("invalid witnessed splice bounds")
    if result["decision"] == "reroute_instructional":
        if result["social_action_visible"] != "yes" or result["behavior_label_supported"] != "yes":
            raise ValueError("instructional reroute requires a visible labeled action")
        if result["authenticity"] not in {"scripted", "animation"}:
            raise ValueError("instructional reroute requires scripted media")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.batch / "manifest.json").read_text())
    if manifest.get("pillar") != "witnessed" or manifest.get("rubric_version") != PROMPT_VERSION:
        raise SystemExit("batch pillar/rubric does not match witnessed_v1")
    items = [item for item in manifest["items"] if item.get("frames")]
    if args.limit:
        items = items[: args.limit]
    out = args.out or (args.batch / "results.jsonl")
    done = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["item_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    if args.dry_run:
        for item in items:
            if not content_for_item(args.batch, item):
                raise AssertionError("empty payload")
        print(json.dumps({"items": len(items), "images": sum(len(x["frames"]) for x in items), "prompt_sha256": prompt_hash()}))
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required; no calls were made")

    from openai import OpenAI

    client = OpenAI()
    with out.open("a") as handle:
        for item in items:
            if item["item_id"] in done:
                continue
            response = client.responses.create(
                model=args.model,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "user", "content": content_for_item(args.batch, item)}],
                reasoning={"effort": "medium"},
                text={"format": {"type": "json_schema", "name": "witnessed_visual_audit", "strict": True, "schema": SCHEMA}},
                max_output_tokens=1000,
                store=False,
            )
            result = json.loads(response.output_text)
            validate_result(result, item)
            usage = response.usage
            record = {
                "item_id": item["item_id"],
                "batch_id": manifest["batch_id"],
                "rubric_version": PROMPT_VERSION,
                "model": args.model,
                "prompt_sha256": prompt_hash(),
                "frame_manifest_sha256": item["frame_manifest_sha256"],
                "pass_index": 0,
                **result,
                "raw_response": result,
                "response_id": response.id,
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "audited_at": time.time(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
