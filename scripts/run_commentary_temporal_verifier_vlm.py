#!/usr/bin/env python3
"""Run a two-stage, label-blind-first verifier over bounded motion pages."""

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


STAGE_A_SYSTEM_TEMPLATE = """You inspect consecutive video frames without being
told what event anyone claims happened. Describe only literal temporal
evidence. Pages are chronological and each page is a {columns}-column by
{rows}-row grid sampled at {fps:g} frames per second. Read left-to-right,
top-to-bottom.

Do not infer an action from captions, logos, narration, a person's identity,
mere proximity, possession in only one frame, aftermath, or the fact that a
camera is pointed at something. A completed action requires a coherent
frame-to-frame transition: actor before, observable action/contact/transfer,
and a compatible after-state. Mark occluded or off-frame whenever the crucial
hand, target, contact, object transfer, road interaction, or result crosses a
camera boundary or is hidden by bodies, doors, blur, inset size, or an edit.

Return exactly one JSON object:
sequence_contains_observable_event: yes|no|uncertain
actor_visible: yes|no|uncertain
target_or_property_visible: yes|no|uncertain
before_action_after_complete: yes|no|uncertain
crucial_action_occluded_or_offframe: yes|no|uncertain
caption_or_narration_needed: yes|no|uncertain
observable_action_kind: physical_contact|property_transfer|dangerous_vehicle_maneuver|situated_interaction|ordinary_activity|no_action|uncertain
literal_before: short literal description
literal_action: short literal description
literal_after: short literal description
evidence: one concise sentence about the ordered frames

Use yes for sequence_contains_observable_event only when the frames themselves
establish a non-ordinary interaction. Situated speech cannot be verified from
silent frames and should be uncertain unless a separate audio pass is used.
{mask_note}"""


STAGE_B_SYSTEM = """You align a fallible source claim to a title-blind literal
frame analysis. Treat the source claim only as a hypothesis. You may not add
actions, objects, targets, intent, ownership, causality, or polarity absent
from the literal analysis. Reject if the crucial action is occluded/off-frame,
if only aftermath or possession is present, if captions/narration are needed,
or if the before-action-after transition is incomplete.

Return exactly one JSON object:
claim_action_in_literal_sequence: yes|no|uncertain
same_actor_action_target: yes|no|uncertain
claim_requires_unseen_intent_or_ownership: yes|no|uncertain
literal_sequence_contradicts_claim: yes|no|uncertain
decision: pass|reject|review
evidence: one concise sentence

Pass requires the literal analysis itself to show the claimed actor-action-
target transition. Use review for genuinely resolvable crop/frame-rate cases,
not to rescue a title with narration."""


STAGE_A_SYSTEM_V3_TEMPLATE = """You inspect consecutive video frames without
being told what event anyone claims happened. Describe only literal temporal
evidence. Pages are chronological and each page is a {columns}-column by
{rows}-row grid sampled at {fps:g} frames per second. Read left-to-right,
top-to-bottom.

Do not infer an action from captions, logos, narration, identity, mere
proximity, possession in only one frame, aftermath, or the fact that a camera
is pointed at something. A completed action requires a coherent transition:
actor before, observable action/contact/transfer, and compatible after-state.
Mark occluded/off-frame whenever the crucial hand, target, contact, transfer,
road interaction, or result crosses a boundary or is hidden by bodies, doors,
blur, inset size, or an edit.

Return exactly one JSON object:
sequence_contains_observable_event: yes|no|uncertain
actor_visible: yes|no|uncertain
target_or_property_visible: yes|no|uncertain
before_action_after_complete: yes|no|uncertain
crucial_action_occluded_or_offframe: yes|no|uncertain
caption_or_narration_needed: yes|no|uncertain
observable_action_kind: physical_contact|property_transfer|dangerous_vehicle_maneuver|situated_interaction|ordinary_activity|no_action|uncertain
literal_actor: short visible description, or none
literal_target: short visible person/object description, or none
literal_before: short literal description
literal_action: short literal description
literal_after: short literal description
literal_result: short visible state change, or none
evidence: one concise sentence about the ordered frames

The literal_action and literal_target fields must explicitly name what pixels
show; flags alone are not evidence. Situated speech cannot be verified from
silent frames. {mask_note}"""


STAGE_B_SYSTEM_V3 = """You align a fallible weak-label claim to a title-blind
literal frame analysis. Separate the visually testable physical core from
semantic qualifiers such as identity, ownership, intent, legal status, motive,
and downstream injury.

Pass only when the supplied Stage-A text itself explicitly names the claimed
core actor-action-target transition and visible result. You may not add a
target or action absent from Stage A. For a theft claim, a visible pickup or
concealment followed by removal/departure may support the physical core even
though ownership or intent comes from the weak label. Generic proximity,
leaning into a vehicle, possession, an edit, aftermath, or ordinary lane
movement cannot support an assault, theft, collision, or other stronger claim.

Return exactly one JSON object:
core_action_in_literal_sequence: yes|no|uncertain
core_target_and_result_in_literal_sequence: yes|no|uncertain
stage_a_text_explicitly_names_core_action: yes|no|uncertain
semantic_qualifiers_require_external_label: yes|no|uncertain
literal_sequence_contradicts_core_action: yes|no|uncertain
decision: pass|reject|review
evidence: one concise sentence quoting or closely paraphrasing only Stage-A fields

Use review only for a resolvable crop/frame-rate issue, not to rescue a source
claim."""


TRI = {"yes", "no", "uncertain"}
ACTION_KINDS = {
    "physical_contact",
    "property_transfer",
    "dangerous_vehicle_maneuver",
    "situated_interaction",
    "ordinary_activity",
    "no_action",
    "uncertain",
}


def stage_a_system(
    row: dict[str, Any],
    policy: str = "strict_exact_v2",
) -> str:
    """Describe the actual storyboard layout and any anti-leakage transform."""
    columns = int(row.get("grid_columns", 8))
    rows = int(row.get("grid_rows", 4))
    fps = float(row.get("fps", 4))
    if columns <= 0 or rows <= 0 or fps <= 0:
        raise ValueError("invalid storyboard layout metadata")
    mask_note = ""
    if row.get("ocr_masked"):
        mask_note = (
            "Flat neutral patches cover OCR-detected text. Treat those patches "
            "only as redactions: do not interpret their shape, location, or "
            "appearance as an object, action, or event clue."
        )
    template = (
        STAGE_A_SYSTEM_V3_TEMPLATE
        if policy == "core_event_v3"
        else STAGE_A_SYSTEM_TEMPLATE
    )
    return template.format(
        columns=columns,
        rows=rows,
        fps=fps,
        mask_note=mask_note,
    )


def rubric_name(
    row: dict[str, Any],
    policy: str = "strict_exact_v2",
) -> str:
    if policy == "core_event_v3":
        return "commentary_temporal_core_event_v3_ocr_masked"
    return (
        "commentary_temporal_verifier_v2_ocr_masked"
        if row.get("ocr_masked")
        else "commentary_temporal_verifier_v1"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response contains no JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def parse_stage_a(text: str) -> dict[str, Any]:
    result = json_object(text)
    expected = {
        "sequence_contains_observable_event",
        "actor_visible",
        "target_or_property_visible",
        "before_action_after_complete",
        "crucial_action_occluded_or_offframe",
        "caption_or_narration_needed",
        "observable_action_kind",
        "literal_before",
        "literal_action",
        "literal_after",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError("stage A schema mismatch")
    for field in expected & {
        "sequence_contains_observable_event",
        "actor_visible",
        "target_or_property_visible",
        "before_action_after_complete",
        "crucial_action_occluded_or_offframe",
        "caption_or_narration_needed",
    }:
        if result[field] not in TRI:
            raise ValueError(f"invalid {field}")
    if result["observable_action_kind"] not in ACTION_KINDS:
        raise ValueError("invalid observable_action_kind")
    for field in ("literal_before", "literal_action", "literal_after", "evidence"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"invalid {field}")
    return result


def parse_stage_a_v3(text: str) -> dict[str, Any]:
    result = json_object(text)
    expected = {
        "sequence_contains_observable_event",
        "actor_visible",
        "target_or_property_visible",
        "before_action_after_complete",
        "crucial_action_occluded_or_offframe",
        "caption_or_narration_needed",
        "observable_action_kind",
        "literal_actor",
        "literal_target",
        "literal_before",
        "literal_action",
        "literal_after",
        "literal_result",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError("stage A v3 schema mismatch")
    for field in {
        "sequence_contains_observable_event",
        "actor_visible",
        "target_or_property_visible",
        "before_action_after_complete",
        "crucial_action_occluded_or_offframe",
        "caption_or_narration_needed",
    }:
        if result[field] not in TRI:
            raise ValueError(f"invalid {field}")
    if result["observable_action_kind"] not in ACTION_KINDS:
        raise ValueError("invalid observable_action_kind")
    for field in (
        "literal_actor",
        "literal_target",
        "literal_before",
        "literal_action",
        "literal_after",
        "literal_result",
        "evidence",
    ):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"invalid {field}")
    return result


def parse_stage_b(text: str) -> dict[str, Any]:
    result = json_object(text)
    expected = {
        "claim_action_in_literal_sequence",
        "same_actor_action_target",
        "claim_requires_unseen_intent_or_ownership",
        "literal_sequence_contradicts_claim",
        "decision",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError("stage B schema mismatch")
    for field in expected & {
        "claim_action_in_literal_sequence",
        "same_actor_action_target",
        "claim_requires_unseen_intent_or_ownership",
        "literal_sequence_contradicts_claim",
    }:
        if result[field] not in TRI:
            raise ValueError(f"invalid {field}")
    if result["decision"] not in {"pass", "reject", "review"}:
        raise ValueError("invalid decision")
    if not isinstance(result["evidence"], str) or not result["evidence"].strip():
        raise ValueError("invalid evidence")
    return result


def parse_stage_b_v3(text: str) -> dict[str, Any]:
    result = json_object(text)
    expected = {
        "core_action_in_literal_sequence",
        "core_target_and_result_in_literal_sequence",
        "stage_a_text_explicitly_names_core_action",
        "semantic_qualifiers_require_external_label",
        "literal_sequence_contradicts_core_action",
        "decision",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError("stage B v3 schema mismatch")
    for field in expected - {"decision", "evidence"}:
        if result[field] not in TRI:
            raise ValueError(f"invalid {field}")
    if result["decision"] not in {"pass", "reject", "review"}:
        raise ValueError("invalid decision")
    if not isinstance(result["evidence"], str) or not result["evidence"].strip():
        raise ValueError("invalid evidence")
    return result


def strict_pass(stage_a: dict[str, Any], stage_b: dict[str, Any]) -> bool:
    return (
        stage_a["sequence_contains_observable_event"] == "yes"
        and stage_a["before_action_after_complete"] == "yes"
        and stage_a["crucial_action_occluded_or_offframe"] == "no"
        and stage_a["caption_or_narration_needed"] == "no"
        and stage_b["claim_action_in_literal_sequence"] == "yes"
        and stage_b["same_actor_action_target"] == "yes"
        and stage_b["claim_requires_unseen_intent_or_ownership"] == "no"
        and stage_b["literal_sequence_contradicts_claim"] == "no"
        and stage_b["decision"] == "pass"
    )


def strict_pass_v3(stage_a: dict[str, Any], stage_b: dict[str, Any]) -> bool:
    return (
        stage_a["sequence_contains_observable_event"] == "yes"
        and stage_a["actor_visible"] == "yes"
        and stage_a["target_or_property_visible"] == "yes"
        and stage_a["before_action_after_complete"] == "yes"
        and stage_a["crucial_action_occluded_or_offframe"] == "no"
        and stage_a["caption_or_narration_needed"] == "no"
        and stage_b["core_action_in_literal_sequence"] == "yes"
        and stage_b["core_target_and_result_in_literal_sequence"] == "yes"
        and stage_b["stage_a_text_explicitly_names_core_action"] == "yes"
        and stage_b["literal_sequence_contradicts_core_action"] == "no"
        and stage_b["decision"] == "pass"
    )


def chat(
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
) -> tuple[str, dict[str, Any] | None]:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 700,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    content = body["choices"][0]["message"].get("content")
    if not content:
        raise ValueError("empty response")
    return content, body.get("usage")


def exception_detail(exc: BaseException) -> str:
    """Preserve a bounded server response body for actionable HTTP failures."""
    detail = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read(4096).decode("utf-8", errors="replace").strip()
        except (OSError, ValueError):
            body = ""
        if body:
            detail += f"; response_body={body}"
    return detail


def request_one(
    endpoint: str,
    model: str,
    row: dict[str, Any],
    manifest_dir: Path,
    timeout: int,
    retries: int,
    policy: str = "strict_exact_v2",
) -> dict[str, Any]:
    rubric = rubric_name(row, policy)
    pages = []
    for value in row["page_paths"]:
        page = Path(value)
        pages.append(page if page.is_absolute() else manifest_dir / page)
    if len(pages) != len(row["page_sha256"]):
        raise ValueError("page/hash length mismatch")
    for page, expected in zip(pages, row["page_sha256"]):
        if sha256(page) != expected:
            raise ValueError(f"page hash mismatch: {page}")
    visual_content = [
        {"type": "image_url", "image_url": {"url": page.resolve().as_uri()}}
        for page in pages
    ]
    visual_content.append(
        {
            "type": "text",
            "text": (
                "Describe the literal ordered transition in these consecutive "
                "frames. No source title or claimed event is provided. "
                + (
                    "Some text regions were redacted before you received them."
                    if row.get("ocr_masked")
                    else ""
                )
            ),
        }
    )
    last_error = ""
    raw_a = raw_b = None
    for attempt in range(retries + 1):
        try:
            raw_a, usage_a = chat(
                endpoint,
                model,
                [
                    {
                        "role": "system",
                        "content": stage_a_system(row, policy),
                    },
                    {"role": "user", "content": visual_content},
                ],
                timeout,
            )
            stage_a = (
                parse_stage_a_v3(raw_a)
                if policy == "core_event_v3"
                else parse_stage_a(raw_a)
            )
            raw_b, usage_b = chat(
                endpoint,
                model,
                [
                    {
                        "role": "system",
                        "content": (
                            STAGE_B_SYSTEM_V3
                            if policy == "core_event_v3"
                            else STAGE_B_SYSTEM
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Fallible source claim: {row['fallible_claim']}\n"
                            "Title-blind literal frame analysis:\n"
                            f"{json.dumps(stage_a, sort_keys=True)}"
                        ),
                    },
                ],
                timeout,
            )
            stage_b = (
                parse_stage_b_v3(raw_b)
                if policy == "core_event_v3"
                else parse_stage_b(raw_b)
            )
            return {
                "audit_index": row["audit_index"],
                "candidate_id": row["candidate_id"],
                "model": model,
                "rubric": rubric,
                "stage_a": stage_a,
                "stage_b": stage_b,
                "strict_pass": (
                    strict_pass_v3(stage_a, stage_b)
                    if policy == "core_event_v3"
                    else strict_pass(stage_a, stage_b)
                ),
                "raw_stage_a": raw_a,
                "raw_stage_b": raw_b,
                "usage": {"stage_a": usage_a, "stage_b": usage_b},
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
            last_error = exception_detail(exc)
            if attempt < retries:
                time.sleep(2**attempt)
    return {
        "audit_index": row["audit_index"],
        "candidate_id": row["candidate_id"],
        "model": model,
        "rubric": rubric,
        "stage_a": None,
        "stage_b": None,
        "strict_pass": False,
        "raw_stage_a": raw_a,
        "raw_stage_b": raw_b,
        "usage": None,
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--policy",
        choices=("strict_exact_v2", "core_event_v3"),
        default="strict_exact_v2",
    )
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    completed = {
        (row["candidate_id"], row["model"], row.get("rubric"))
        for row in read_jsonl(args.out)
        if row.get("error") is None and row.get("stage_b") is not None
    } if args.out.exists() else set()
    pending = [
        row for row in rows
        if (
            row["candidate_id"],
            args.model,
            rubric_name(row, args.policy),
        ) not in completed
    ]
    produced: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                request_one,
                args.endpoint,
                args.model,
                row,
                args.manifest.parent,
                args.timeout,
                args.retries,
                args.policy,
            ): row
            for row in pending
        }
        for count, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                produced[int(row["audit_index"])] = future.result()
            except Exception as exc:
                produced[int(row["audit_index"])] = {
                    "audit_index": row["audit_index"],
                    "candidate_id": row["candidate_id"],
                    "model": args.model,
                    "rubric": rubric_name(row, args.policy),
                    "stage_a": None,
                    "stage_b": None,
                    "strict_pass": False,
                    "raw_stage_a": None,
                    "raw_stage_b": None,
                    "usage": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if count % 10 == 0 or count == len(futures):
                print(f"{count}/{len(futures)} completed", flush=True)
    existing = read_jsonl(args.out) if args.out.exists() else []
    args.out.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in existing + [produced[index] for index in sorted(produced)]
        )
    )
    failures = sum(row["error"] is not None for row in produced.values())
    print(json.dumps({"requested": len(pending), "failures": failures}))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
