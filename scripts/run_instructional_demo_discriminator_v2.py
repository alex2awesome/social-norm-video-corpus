#!/usr/bin/env python3
"""Classify visual demonstrations without seeing the proposed norm label."""

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
        _json_object, base_record, load_jsonl, sha256, successful_keys,
    )
    from scripts.run_instructional_blind_episode_v1 import quotes_by_item
    from scripts.run_instructional_v15_structure_vlm import normalize_manifest_row
except ModuleNotFoundError:
    from run_open_vlm_v10_storyboards import (  # type: ignore[no-redef]
        _json_object, base_record, load_jsonl, sha256, successful_keys,
    )
    from run_instructional_blind_episode_v1 import quotes_by_item  # type: ignore[no-redef]
    from run_instructional_v15_structure_vlm import normalize_manifest_row  # type: ignore[no-redef]


SYSTEM_DEMO_DISCRIMINATOR = """You decide whether an instructional clip
VISUALLY DEPICTS a concrete social behavior. You receive a dense chronological
storyboard and only the transcript quotes from that saved interval. You do not
receive the title, category, proposed norm, polarity, explanation, or search
query. Do not infer a lesson that is not visible.

A valid demonstration may be live, captured reality, roleplay, film/TV,
animation, puppets, synthetic imagery, a clear static tableau, a phone call, or
one person visibly demonstrating a recognizable social convention. The event
need not be complete. Situated speech such as an insult, refusal, request, or
correction is itself social behavior when it is genuinely addressed within the
depicted episode. Direct care or assistance is also social behavior.

These are NOT visual demonstrations: interviews; red-carpet or studio
conversation programs; lectures, panels, sermons, and direct-to-camera advice;
presenters alternating with disconnected b-roll; generic conversation without
a concrete social act; and technical instruction in exercise, medicine,
software, music, tools, cooking, or other mechanics. Touching a student while
teaching posture is technical, not social. Discussing or reacting to an
off-screen event is not depicting that event.

Solo actions count only when the pixels visibly demonstrate a recognizable
social convention such as table manners, greeting, queuing, door etiquette, or
public-space conduct. Ordinary grooming, exercise, performance, and self-care
do not count merely because narration supplies a moral. Animation, puppets, and
static illustration are not penalized if they depict the behavior clearly.

Inspect the whole storyboard. Distinguish participants actually sharing an
episode from people appearing in separate presenter/b-roll shots. Return
exactly one JSON object with:

depiction_medium: live_action|film_tv|animation|puppets|synthetic_or_static|screen_content|other|none|unclear
depiction_mode: captured_or_fictional_episode|roleplay_episode|solo_etiquette_demonstration|static_behavior_tableau|interview_or_conversation_program|lecture_panel_or_presenter|presenter_with_disconnected_broll|technical_skill_demonstration|generic_activity_or_self_care|text_graphic_or_gameplay|other|none|unclear
behavior_source: visibly_performed|visibly_static_depiction|only_spoken_or_described|none|unclear
social_act_kind: interpersonal_physical_action|situated_social_speech|caregiving_or_assistance|shared_public_conduct|solo_social_convention|technical_non_social_action|generic_conversation|none|unclear
recipient_grounding: visible_recipient|visible_bystander_or_shared_public|offscreen_recipient_same_episode|solo_convention|none|unclear
episode_continuity: same_episode|clear_single_tableau|disconnected_or_presenter_broll|none|unclear
interview_or_presentation: yes|no|unclear
technical_skill: yes|no|unclear
concrete_visible_behavior: one short literal description, or none
evidence: one short sentence explaining the classifications

Keep free text under 40 words. Return JSON only."""


ENUMS = {
    "depiction_medium": {
        "live_action", "film_tv", "animation", "puppets",
        "synthetic_or_static", "screen_content", "other", "none", "unclear",
    },
    "depiction_mode": {
        "captured_or_fictional_episode", "roleplay_episode",
        "solo_etiquette_demonstration", "static_behavior_tableau",
        "interview_or_conversation_program", "lecture_panel_or_presenter",
        "presenter_with_disconnected_broll", "technical_skill_demonstration",
        "generic_activity_or_self_care", "text_graphic_or_gameplay", "other",
        "none", "unclear",
    },
    "behavior_source": {
        "visibly_performed", "visibly_static_depiction",
        "only_spoken_or_described", "none", "unclear",
    },
    "social_act_kind": {
        "interpersonal_physical_action", "situated_social_speech",
        "caregiving_or_assistance", "shared_public_conduct",
        "solo_social_convention", "technical_non_social_action",
        "generic_conversation", "none", "unclear",
    },
    "recipient_grounding": {
        "visible_recipient", "visible_bystander_or_shared_public",
        "offscreen_recipient_same_episode", "solo_convention", "none", "unclear",
    },
    "episode_continuity": {
        "same_episode", "clear_single_tableau",
        "disconnected_or_presenter_broll", "none", "unclear",
    },
    "interview_or_presentation": {"yes", "no", "unclear"},
    "technical_skill": {"yes", "no", "unclear"},
}
TEXT_FIELDS = {"concrete_visible_behavior", "evidence"}
ALLOWED_MODES = {
    "captured_or_fictional_episode", "roleplay_episode",
    "solo_etiquette_demonstration", "static_behavior_tableau",
}
ALLOWED_ACTS = {
    "interpersonal_physical_action", "situated_social_speech",
    "caregiving_or_assistance", "shared_public_conduct",
    "solo_social_convention",
}
ALLOWED_GROUNDING = {
    "visible_recipient", "visible_bystander_or_shared_public",
    "offscreen_recipient_same_episode", "solo_convention",
}


def derive_demo_candidate(result: dict[str, Any]) -> bool:
    return (
        result["depiction_mode"] in ALLOWED_MODES
        and result["behavior_source"]
        in {"visibly_performed", "visibly_static_depiction"}
        and result["social_act_kind"] in ALLOWED_ACTS
        and result["recipient_grounding"] in ALLOWED_GROUNDING
        and result["episode_continuity"]
        in {"same_episode", "clear_single_tableau"}
        and result["interview_or_presentation"] == "no"
        and result["technical_skill"] == "no"
    )


def parse_demo_discriminator(text: str) -> dict[str, Any]:
    result = _json_object(text)
    expected = set(ENUMS) | TEXT_FIELDS
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    for field, values in ENUMS.items():
        if result[field] not in values:
            raise ValueError(f"invalid {field}: {result[field]!r}")
    for field in TEXT_FIELDS:
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"invalid {field}")
    result["demo_candidate"] = derive_demo_candidate(result)
    return result


def request_one(
    endpoint: str, model: str, row: dict[str, Any], timeout: int, retries: int,
) -> dict[str, Any]:
    sheet = Path(row["sheet_path"])
    if not sheet.is_absolute():
        sheet = Path(row["_manifest_dir"]) / sheet
    if sha256(sheet) != row["sheet_sha256"]:
        raise ValueError(f"storyboard hash mismatch: {row['item_id']}")
    quotes = row["_quotes"]
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 900,
        "messages": [
            {"role": "system", "content": SYSTEM_DEMO_DISCRIMINATOR},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": sheet.resolve().as_uri()}},
                {"type": "text", "text": "Interval quotes only:\n" + json.dumps(quotes, sort_keys=True)},
            ]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode()
    last_error = ""
    last_content = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions", data=encoded,
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
                **base_record(row, model), "rubric": "demo_discriminator_v2",
                "quote_context": quotes,
                "result": parse_demo_discriminator(content),
                "raw_response": content, "usage": body.get("usage"), "error": None,
            }
        except urllib.error.HTTPError as exc:
            last_error = f"HTTPError {exc.code}: {exc.read().decode(errors='replace')[:2000]}"
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(2**attempt)
    return {
        **base_record(row, model), "rubric": "demo_discriminator_v2",
        "quote_context": quotes, "result": None, "raw_response": last_content,
        "usage": None, "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sheet-root", type=Path)
    args = parser.parse_args()
    rows = [
        normalize_manifest_row(row, args.manifest.parent, "instructional")
        for row in load_jsonl(args.manifest)
    ]
    if args.sheet_root is not None:
        rows = [
            {**row, "sheet_path": str(args.sheet_root / Path(row["sheet_path"]).name)}
            for row in rows
        ]
    quotes = quotes_by_item(load_jsonl(args.semantic_manifest))
    if {str(row["item_id"]) for row in rows} != set(quotes):
        raise ValueError("storyboard and semantic cohorts differ")
    rows = [{**row, "_quotes": quotes[str(row["item_id"])]} for row in rows]
    completed = successful_keys(load_jsonl(args.out)) if args.out.exists() else set()
    pending = [row for row in rows if (row["item_id"], args.model) not in completed]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                request_one, args.endpoint, args.model, row, args.timeout, args.retries,
            ): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{index}/{len(futures)} {record['item_id']} error={record['error']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
