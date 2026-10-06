#!/usr/bin/env python3
"""Find temporally performed instructional demos with a conservative VLM critic."""

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


SYSTEM_TEMPORAL_CRITIC = """You are a conservative temporal visual-evidence
critic. Decide whether any continuous subsegment of an instructional clip
actually performs a specific social behavior. You see 36 frames in time order
and interval quotes, but no title, norm, category, polarity, explanation, or
query. Inspect every frame: a short valid segment counts even when the rest is
a presenter or title card.

Accept any medium when behavior unfolds: captured reality, film/TV, roleplay,
animation, puppets, screen action, or coordination among vehicles or other
agents. A situated utterance counts when a real same-episode recipient is
visible, or roleplay visually establishes an off-screen recipient through
staging rather than words alone. A visible before/act/after change, object
transfer, recipient response, turn consequence, or coordinated trajectory is
temporal evidence.

Fail these lookalikes:
- A presenter, interview, panel, host exchange, or sample phrase to camera.
- Generic conversation, teaching, walking, training, shopping, eating, or
  gesturing without one exact socially interpretable act.
- Static art, storybook pages, stock cutouts, diagrams, or captions that state
  an action which does not unfold. Do not convert a pose into motion.
- Disconnected montage or B-roll joined only by narration.
- Technical procedures, fitness, tools, driving mechanics, cooking, music,
  software, or generic self-care unless the visible act itself is a social
  convention.

Do not infer motion, a recipient, or a consequence from quotes. First locate
the best candidate subsegment using zero-based frame numbers 0 through 35.
Return exactly one JSON object with:
segment_kind: continuous_episode|roleplay_or_fiction|animated_episode|screen_action_episode|public_coordination_episode|clear_behavioral_tableau|presenter_or_interview|disconnected_montage|technical_or_generic|static_or_caption_assertion|none|unclear
segment_start_frame: integer 0..35, or -1
segment_end_frame: integer 0..35, or -1
recipient_configuration: visible_recipient_or_affected_party|visually_established_offscreen_recipient|coordinating_agents_or_vehicles|recognizable_solo_social_convention|no_recipient_or_target|unclear
temporal_act_evidence: visible_before_act_after|visible_act_and_response|visible_state_change|clear_static_tableau_no_motion|only_generic_turn_taking|only_quote_or_caption|none|unclear
specific_social_act: yes|no|unclear
generic_conversation_or_activity: yes|no|unclear
static_meaning_only: yes|no|unclear
presenter_or_program: yes|no|unclear
disconnected_montage: yes|no|unclear
technical_or_nonsocial: yes|no|unclear
performed_act: one literal short phrase, or none
before_evidence: one short visual observation, or none
during_evidence: one short visual observation, or none
after_evidence: one short visual observation, or none
evidence: one sentence explaining why the candidate passes or fails

Keep free text under 60 words total. Return JSON only."""


ENUMS = {
    "segment_kind": {
        "continuous_episode", "roleplay_or_fiction", "animated_episode",
        "screen_action_episode", "public_coordination_episode",
        "clear_behavioral_tableau", "presenter_or_interview",
        "disconnected_montage", "technical_or_generic",
        "static_or_caption_assertion", "none", "unclear",
    },
    "recipient_configuration": {
        "visible_recipient_or_affected_party",
        "visually_established_offscreen_recipient",
        "coordinating_agents_or_vehicles", "recognizable_solo_social_convention",
        "no_recipient_or_target", "unclear",
    },
    "temporal_act_evidence": {
        "visible_before_act_after", "visible_act_and_response",
        "visible_state_change", "clear_static_tableau_no_motion",
        "only_generic_turn_taking", "only_quote_or_caption", "none", "unclear",
    },
    "specific_social_act": {"yes", "no", "unclear"},
    "generic_conversation_or_activity": {"yes", "no", "unclear"},
    "static_meaning_only": {"yes", "no", "unclear"},
    "presenter_or_program": {"yes", "no", "unclear"},
    "disconnected_montage": {"yes", "no", "unclear"},
    "technical_or_nonsocial": {"yes", "no", "unclear"},
}
TEXT_FIELDS = {
    "performed_act", "before_evidence", "during_evidence", "after_evidence",
    "evidence",
}
INTEGER_FIELDS = {"segment_start_frame", "segment_end_frame"}


def derive_demo_candidate(result: dict[str, Any]) -> bool:
    return (
        result["segment_kind"] in {
            "continuous_episode", "roleplay_or_fiction", "animated_episode",
            "screen_action_episode", "public_coordination_episode",
        }
        and result["recipient_configuration"] in {
            "visible_recipient_or_affected_party",
            "visually_established_offscreen_recipient",
            "coordinating_agents_or_vehicles",
            "recognizable_solo_social_convention",
        }
        and result["temporal_act_evidence"] in {
            "visible_before_act_after", "visible_act_and_response",
            "visible_state_change",
        }
        and result["specific_social_act"] == "yes"
        and result["generic_conversation_or_activity"] == "no"
        and result["static_meaning_only"] == "no"
        and result["presenter_or_program"] == "no"
        and result["disconnected_montage"] == "no"
        and result["technical_or_nonsocial"] == "no"
        and 0 <= result["segment_start_frame"] <= result["segment_end_frame"] <= 35
    )


def parse_temporal_critic(text: str) -> dict[str, Any]:
    result = _json_object(text)
    expected = set(ENUMS) | TEXT_FIELDS | INTEGER_FIELDS
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    for field, allowed in ENUMS.items():
        if result[field] not in allowed:
            raise ValueError(f"invalid {field}: {result[field]!r}")
    for field in TEXT_FIELDS:
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"invalid {field}")
    for field in INTEGER_FIELDS:
        if not isinstance(result[field], int) or isinstance(result[field], bool):
            raise ValueError(f"invalid {field}")
        if result[field] < -1 or result[field] > 35:
            raise ValueError(f"invalid {field}")
    if (result["segment_start_frame"] == -1) != (result["segment_end_frame"] == -1):
        raise ValueError("segment frame bounds must both be -1 or both nonnegative")
    if (
        result["segment_start_frame"] >= 0
        and result["segment_start_frame"] > result["segment_end_frame"]
    ):
        raise ValueError("segment start exceeds end")
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
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1100,
        "messages": [
            {"role": "system", "content": SYSTEM_TEMPORAL_CRITIC},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": sheet.resolve().as_uri()}},
                {"type": "text", "text": "Interval quotes only:\n" + json.dumps(row["_quotes"], sort_keys=True)},
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
                **base_record(row, model), "rubric": "temporal_critic_v4",
                "quote_context": row["_quotes"],
                "result": parse_temporal_critic(content), "raw_response": content,
                "usage": body.get("usage"), "error": None,
            }
        except urllib.error.HTTPError as exc:
            last_error = f"HTTPError {exc.code}: {exc.read().decode(errors='replace')[:2000]}"
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(2**attempt)
    return {
        **base_record(row, model), "rubric": "temporal_critic_v4",
        "quote_context": row["_quotes"], "result": None,
        "raw_response": last_content, "usage": None, "error": last_error,
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
            pool.submit(request_one, args.endpoint, args.model, row, args.timeout, args.retries): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(f"{index}/{len(futures)} {record['item_id']} error={record['error']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
