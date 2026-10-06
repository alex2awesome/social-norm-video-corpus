#!/usr/bin/env python3
"""Audit an exported frame batch with GPT-5.6 structured vision judgments."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


PROMPT_VERSION = "instructional_v4"
SYSTEM_PROMPT = """You are the final visual QA auditor for a social-norm video dataset.
Judge the supplied chronological frames as evidence from ONE extracted clip.

A valid instructional demo visibly depicts an agent performing, attempting, or
refraining from an observable behavior in a social context. ANY medium is valid:
live action, animation, puppets, dolls, or a visible on-screen social exchange.
Talking heads, lectures, static slides, definitions, generic B-roll, and merely
mentioning a norm are not demos. A speech act such as apologizing, insulting, or
refusing can be a demo when the frames show the participants in an active social
exchange and the aligned transcript grounds what is said. Do not accept narration
over unrelated B-roll as a speech-act demo. Technical procedures, animal training,
medical advice, generic physical safety, and self-improvement are not social norms
unless the visible action is governed by an interpersonal or shared-public expectation.

The proposed norm and explanation are weak labels, not facts. Decide whether the
combined visual and aligned-dialogue evidence supports them. Do not infer an unseen
scene from a title, explanation, or narrator statement. Use uncertain when the
sampled evidence cannot establish the answer.

An accepted training input must still depict and support a concrete social behavior
after every required repair is applied. Repairs are: trim a label/explanation-only
interval; strip explanatory audio when pixels alone prove the behavior; relabel a
wrong or malformed norm when a different concrete norm is visibly grounded; or
relabel a wrong polarity. Do not use relabeling to rescue generic B-roll. Do not
strip audio when in-character dialogue is needed to identify the behavior. Provide
specific normalized behavior and norm for every accepted item.

Use accept only for a pure demo requiring no repair. Use accept_with_repairs only
with a nonempty, sufficient repair list. Visible label leakage must be completely
removed by a clean trim; explanatory narration must be completely removed by a
clean trim or strip_explanatory_audio. In-character dialogue is behavior, not
narration leakage. Ordinary dialogue subtitles are not label leakage. For proposed
explanation polarity, judge the scene's actual polarity and mark
proposed_polarity_matches not_applicable. Identify the frame indices that prove the
post-repair behavior."""


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_social_norm": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "visual_demo_present": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "medium": {
            "type": "string",
            "enum": [
                "live_action",
                "animation",
                "puppet_or_doll",
                "screen_social_exchange",
                "talking_head",
                "lecture",
                "static_graphic",
                "broll",
                "technical_demo",
                "animal_video",
                "mixed",
                "uncertain",
            ],
        },
        "clip_composition": {
            "type": "string",
            "enum": [
                "pure_demo",
                "mixed_demo_and_explanation",
                "explanation_only",
                "talking_head_only",
                "unrelated",
                "uncertain",
            ],
        },
        "norm_supported": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "polarity_supported": {
            "type": "string",
            "enum": ["violation", "correct", "contrast", "not_applicable", "uncertain"],
        },
        "proposed_polarity_matches": {
            "type": "string",
            "enum": ["yes", "no", "uncertain", "not_applicable"],
        },
        "explanation_supports_norm": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "dialogue_grounded": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "visual_norm_supported_without_audio": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
        "narration_leak": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "label_leak_visible": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "off_topic": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "decision": {
            "type": "string",
            "enum": ["accept", "accept_with_repairs", "reject", "uncertain"],
        },
        "required_repairs": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "trim",
                    "strip_explanatory_audio",
                    "relabel_norm",
                    "relabel_polarity",
                ],
            },
            "uniqueItems": True,
        },
        "trim_start_sec": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "trim_end_sec": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "normalized_behavior": {"type": "string"},
        "normalized_norm": {"type": "string"},
        "rejection_reasons": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "no_visual_demo",
                    "not_social_norm",
                    "norm_mismatch",
                    "polarity_mismatch",
                    "off_topic",
                    "talking_head_or_lecture",
                    "static_or_broll",
                    "technical_or_procedural",
                    "insufficient_temporal_evidence",
                    "label_leak",
                    "other",
                ],
            },
        },
        "evidence_frames": {"type": "array", "items": {"type": "integer"}},
        "description": {"type": "string"},
    },
    "required": [
        "is_social_norm",
        "visual_demo_present",
        "medium",
        "clip_composition",
        "norm_supported",
        "polarity_supported",
        "proposed_polarity_matches",
        "explanation_supports_norm",
        "dialogue_grounded",
        "visual_norm_supported_without_audio",
        "narration_leak",
        "label_leak_visible",
        "off_topic",
        "decision",
        "required_repairs",
        "trim_start_sec",
        "trim_end_sec",
        "normalized_behavior",
        "normalized_norm",
        "rejection_reasons",
        "evidence_frames",
        "description",
    ],
    "additionalProperties": False,
}


def data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def prompt_hash() -> str:
    value = {"version": PROMPT_VERSION, "system": SYSTEM_PROMPT, "schema": SCHEMA}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def content_for_item(root: Path, item: dict[str, Any]) -> list[dict[str, Any]]:
    context = {
        "title": item.get("title"),
        "category": item.get("category"),
        "proposed_polarity": item.get("polarity"),
        "proposed_norm": item.get("norm"),
        "start_quote": item.get("start_quote"),
        "end_quote": item.get("end_quote"),
        "educator_explanation": item.get("explanation"),
        "aligned_transcript": item.get("aligned_transcript") or [],
        "duration_seconds": item.get("duration"),
        "frames": [
            {"frame_index": frame["frame_index"], "timestamp": frame["timestamp"]}
            for frame in item["frames"]
        ],
    }
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "Weak-label context and chronological frame map:\n" + json.dumps(context, ensure_ascii=False),
        }
    ]
    for frame in item["frames"]:
        content.append(
            {
                "type": "input_text",
                "text": f"Frame {frame['frame_index']} at {frame['timestamp']:.3f}s:",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": data_url(root / frame["path"]),
                "detail": "high",
            }
        )
    return content


def validate_result(result: dict[str, Any], item: dict[str, Any]) -> None:
    missing = set(SCHEMA["required"]) - set(result)
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    valid_frames = {frame["frame_index"] for frame in item["frames"]}
    if not set(result["evidence_frames"]).issubset(valid_frames):
        raise ValueError("evidence_frames contains an invalid index")
    accepted = result["decision"] in {"accept", "accept_with_repairs"}
    if accepted and not (
        result["is_social_norm"] == "yes"
        and result["visual_demo_present"] == "yes"
        and result["polarity_supported"] in {"violation", "correct", "contrast"}
        and result["off_topic"] == "no"
    ):
        raise ValueError("accepted decision violates rubric invariants")
    if accepted and (
        not result["normalized_behavior"].strip() or not result["normalized_norm"].strip()
    ):
        raise ValueError("accepted decision requires normalized behavior and norm")
    repairs = set(result["required_repairs"])
    if len(repairs) != len(result["required_repairs"]):
        raise ValueError("required_repairs contains duplicates")
    proposed = str(item.get("polarity") or "").lower()
    if accepted and proposed in {"violation", "correct", "contrast"}:
        if "relabel_polarity" not in repairs and result["proposed_polarity_matches"] != "yes":
            raise ValueError("accepted non-explanation item requires matching proposed polarity")
        if "relabel_polarity" in repairs and result["proposed_polarity_matches"] != "no":
            raise ValueError("polarity relabel requires proposed_polarity_matches=no")
    if result["decision"] == "accept" and result["label_leak_visible"] != "no":
        raise ValueError("accept requires label_leak_visible=no")
    if result["decision"] == "accept" and result["narration_leak"] != "no":
        raise ValueError("accept requires narration_leak=no")
    if result["decision"] == "accept" and result["clip_composition"] != "pure_demo":
        raise ValueError("accept requires clip_composition=pure_demo")
    if result["decision"] == "accept":
        if repairs:
            raise ValueError("accept cannot require repairs")
        if result["norm_supported"] != "yes":
            raise ValueError("accept requires the proposed norm to be supported")
    if result["decision"] == "accept_with_repairs":
        if not repairs:
            raise ValueError("accept_with_repairs requires at least one repair")
        if "relabel_norm" in repairs:
            if result["norm_supported"] != "no":
                raise ValueError("norm relabel requires norm_supported=no")
        elif result["norm_supported"] != "yes":
            raise ValueError("non-relabel acceptance requires the proposed norm")
        if "strip_explanatory_audio" in repairs:
            if result["narration_leak"] != "yes":
                raise ValueError("audio stripping requires narration_leak=yes")
            if result["visual_norm_supported_without_audio"] != "yes":
                raise ValueError("audio stripping requires pixel-only norm support")
            if result["dialogue_grounded"] == "yes":
                raise ValueError("cannot strip audio needed to ground in-character dialogue")
        elif result["narration_leak"] == "yes" and "trim" not in repairs:
            raise ValueError("narration leakage is not removed")
        if result["label_leak_visible"] == "yes" and "trim" not in repairs:
            raise ValueError("visible label leakage is not removed")
    if "trim" in repairs:
        start = result["trim_start_sec"]
        end = result["trim_end_sec"]
        duration = float(item["duration"])
        has_leak = result["label_leak_visible"] == "yes" or result["narration_leak"] == "yes"
        if not has_leak or start is None or end is None:
            raise ValueError("trim repair requires a label leak and trim bounds")
        if result["clip_composition"] != "mixed_demo_and_explanation":
            raise ValueError("trim repair requires mixed_demo_and_explanation")
        if not 0 <= float(start) < float(end) <= duration:
            raise ValueError("trim bounds fall outside the clip")
    elif result["trim_start_sec"] is not None or result["trim_end_sec"] is not None:
        raise ValueError("trim bounds require the trim repair")
    if not accepted and repairs:
        raise ValueError("non-accepted decisions cannot prescribe repairs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.batch / "manifest.json").read_text())
    if manifest["rubric_version"] != PROMPT_VERSION:
        raise SystemExit(
            f"batch rubric {manifest['rubric_version']!r} does not match runner {PROMPT_VERSION!r}"
        )
    items = [item for item in manifest["items"] if item.get("frames")]
    if args.limit:
        items = items[: args.limit]
    out = args.out or (args.batch / "results.jsonl")
    done: set[str] = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["item_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    if args.dry_run:
        total_images = 0
        total_bytes = 0
        for item in items:
            payload = content_for_item(args.batch, item)
            total_images += len(item["frames"])
            total_bytes += sum((args.batch / frame["path"]).stat().st_size for frame in item["frames"])
            if not payload:
                raise AssertionError("empty payload")
        print(json.dumps({"items": len(items), "images": total_images, "bytes": total_bytes, "prompt_sha256": prompt_hash()}))
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
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "instructional_visual_audit",
                        "strict": True,
                        "schema": SCHEMA,
                    }
                },
                max_output_tokens=900,
                store=False,
            )
            result = json.loads(response.output_text)
            validate_result(result, item)
            usage = response.usage
            record = {
                "item_id": item["item_id"],
                "batch_id": manifest["batch_id"],
                "rubric_version": manifest["rubric_version"],
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
