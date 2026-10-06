#!/usr/bin/env python3
"""Run a label-blind participant/action structure audit on storyboards."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from scripts.run_open_vlm_v10_storyboards import (
        _json_object,
        base_record,
        load_jsonl,
        sha256,
        successful_keys,
    )
except ModuleNotFoundError:
    from run_open_vlm_v10_storyboards import (  # type: ignore[no-redef]
        _json_object,
        base_record,
        load_jsonl,
        sha256,
        successful_keys,
    )


SYSTEM_V15S = """You are a LABEL-BLIND structural auditor of a dense temporal
video storyboard. You receive no title, transcript, proposed norm, category,
polarity, or explanation. Never invent them.

Decide only whether the frames visibly establish a SPECIFIC performed social
act with an affected person or social group. The act can be physical or a
specific situated utterance. Live action, role-play, film, animation, puppets,
screen dialogue, illustrated stories, and explicit static social depictions
can all qualify.

A specific situated utterance qualifies only when its actual content is visible
in subtitles, speech bubbles, on-screen dialogue, or an otherwise unambiguous
action/reaction sequence. Two people merely talking, smiling, gesturing,
interviewing, counseling, dining, or sitting together is generic conversation,
not a demonstrated act.

Fail these hallucination traps:
- a presenter or interviewee speaks to the camera;
- a person reports, remembers, explains, or reacts to an off-screen event;
- narration captions state what someone did while the frames show only the
  narrator, affected person, aftermath, or unrelated B-roll;
- one model could invent a second participant from a background, location,
  subtitle, or later portrait;
- an alleged harassment, retaliation, discrimination, insult, refusal, or
  other causative act is only discussed rather than performed;
- generic conversation whose social meaning would require hidden metadata.

An affected party may appear in different ordered frames of the same depicted
episode, but must participate in or react to the specific act. A shared room,
audience, field, office, restaurant, or public location alone is not an
affected party.

Read frames left-to-right then top-to-bottom. Use literal evidence only.
Return exactly one JSON object with:
visible_people_or_characters: zero|one|multiple|uncertain
interaction_structure: specific_action_with_target|specific_situated_utterance_with_target|explicit_static_social_depiction|generic_conversation|direct_to_camera_or_presenter|reported_or_discussed_offscreen_event|ordinary_or_non_social_action|uncertain
causative_social_act_visible: yes|no|uncertain
affected_party_in_depicted_event: yes|no|uncertain
action_specificity: specific|generic|uncertain
target_is_person_or_social_group: yes|no|uncertain
only_report_or_advice: yes|no|uncertain
literal_actor: short literal description or none
literal_action_or_utterance: short literal description or none
literal_affected_party: short literal description or none
structural_demo_pass: yes|no|uncertain
evidence: one short sentence naming the visible act and target or the missing
requirement

structural_demo_pass=yes only if causative_social_act_visible=yes,
affected_party_in_depicted_event=yes, action_specificity=specific,
target_is_person_or_social_group=yes, only_report_or_advice=no, and
interaction_structure is one of the first three specific structures.
Otherwise fail closed."""


YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}
PEOPLE = {"zero", "one", "multiple", "uncertain"}
STRUCTURES = {
    "specific_action_with_target",
    "specific_situated_utterance_with_target",
    "explicit_static_social_depiction",
    "generic_conversation",
    "direct_to_camera_or_presenter",
    "reported_or_discussed_offscreen_event",
    "ordinary_or_non_social_action",
    "uncertain",
}
SPECIFICITY = {"specific", "generic", "uncertain"}
QUALIFYING_STRUCTURES = {
    "specific_action_with_target",
    "specific_situated_utterance_with_target",
    "explicit_static_social_depiction",
}


def normalize_manifest_row(
    row: dict[str, Any],
    manifest_dir: Path,
    default_pillar: str,
) -> dict[str, Any]:
    """Support both semantic manifests and intentionally opaque blind packets."""
    normalized = dict(row)
    opaque_id = normalized.get("candidate_id")
    normalized.setdefault("item_id", opaque_id)
    if not normalized.get("item_id"):
        raise ValueError("manifest row needs item_id or candidate_id")
    normalized.setdefault("uid", opaque_id or normalized["item_id"])
    normalized.setdefault("pillar", default_pillar)
    normalized.setdefault("frame_count", 36)
    normalized["_manifest_dir"] = str(manifest_dir)
    return normalized


def parse_v15s(text: str) -> dict[str, Any]:
    result = _json_object(text)
    expected = {
        "visible_people_or_characters",
        "interaction_structure",
        "causative_social_act_visible",
        "affected_party_in_depicted_event",
        "action_specificity",
        "target_is_person_or_social_group",
        "only_report_or_advice",
        "literal_actor",
        "literal_action_or_utterance",
        "literal_affected_party",
        "structural_demo_pass",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    if result["visible_people_or_characters"] not in PEOPLE:
        raise ValueError("invalid visible_people_or_characters")
    if result["interaction_structure"] not in STRUCTURES:
        raise ValueError("invalid interaction_structure")
    if result["action_specificity"] not in SPECIFICITY:
        raise ValueError("invalid action_specificity")
    atomic = (
        "causative_social_act_visible",
        "affected_party_in_depicted_event",
        "target_is_person_or_social_group",
        "only_report_or_advice",
        "structural_demo_pass",
    )
    if any(result[key] not in YES_NO_UNCERTAIN for key in atomic):
        raise ValueError("invalid yes/no/uncertain field")
    strings = (
        "literal_actor",
        "literal_action_or_utterance",
        "literal_affected_party",
        "evidence",
    )
    if any(
        not isinstance(result[key], str) or not result[key].strip()
        for key in strings
    ):
        raise ValueError("literal fields and evidence must be non-empty strings")
    strict_positive = (
        result["causative_social_act_visible"] == "yes"
        and result["affected_party_in_depicted_event"] == "yes"
        and result["action_specificity"] == "specific"
        and result["target_is_person_or_social_group"] == "yes"
        and result["only_report_or_advice"] == "no"
        and result["interaction_structure"] in QUALIFYING_STRUCTURES
    )
    if result["structural_demo_pass"] == "yes" and not strict_positive:
        result["structural_demo_pass_raw"] = "yes"
        result["structural_demo_pass"] = "uncertain"
        result["consistency_repair"] = "inconsistent_positive_to_uncertain"
    return result


def v15_base_record(row: dict[str, Any], model: str) -> dict[str, Any]:
    record = base_record(row, model)
    record["rubric"] = "v15s_label_blind_structure"
    return record


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
        raise ValueError(f"storyboard hash mismatch: {row['item_id']}")
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 600,
        "messages": [
            {"role": "system", "content": SYSTEM_V15S},
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
                            "Audit the complete ordered storyboard. Distinguish "
                            "a performed act from generic talk or a report."
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
                **v15_base_record(row, model),
                "result": parse_v15s(content),
                "raw_response": content,
                "usage": body.get("usage"),
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
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
        **v15_base_record(row, model),
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8273/v1")
    parser.add_argument("--model", default="gemma-3-27b-it")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--default-pillar", default="instructional")
    args = parser.parse_args()

    rows = [
        normalize_manifest_row(
            row,
            args.manifest.parent,
            args.default_pillar,
        )
        for row in load_jsonl(args.manifest)
    ]
    completed = (
        successful_keys(load_jsonl(args.out)) if args.out.exists() else set()
    )
    pending = [
        row for row in rows if (row["item_id"], args.model) not in completed
    ]
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
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    **v15_base_record(row, args.model),
                    "result": None,
                    "raw_response": None,
                    "usage": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
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
