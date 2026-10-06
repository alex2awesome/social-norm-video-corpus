#!/usr/bin/env python3
"""Adjudicate visual instructional demos using failure-specific atomic checks."""

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


SYSTEM_DEMO_ADJUDICATOR = """You are a conservative visual-evidence
adjudicator. Decide whether an instructional clip itself DEPICTS a specific
social behavior that a learner could identify from the saved visual interval.
You see a dense chronological storyboard and interval transcript quotes, but
no title, proposed norm, category, polarity, explanation, or query.

Accept any medium: captured reality, film/TV, roleplay, animation, puppets,
synthetic illustration, a clear behavioral tableau, or a recognizable solo
social convention. A situated request, insult, refusal, apology, correction,
or other speech act counts when genuine same-episode participants are visibly
established. Interpersonal caregiving counts even when it includes a medical
action. Reality-show or educator roleplay counts if behavior is actually acted.

Fail these common lookalikes:
- A lecturer, host, interviewee, panel, or advice giver merely speaking about
  behavior. Two presenters talking to each other are still presenters.
- A sample phrase addressed to the camera or an inferred caller. Do not invent
  an off-screen recipient from words alone.
- Disconnected stock footage or montage. People walking, eating, shopping, or
  handling a prop is not one episode merely because narration links the shots.
- A diagram, text slide, quiz, or static group pose that names a behavior but
  does not depict it.
- Generic conversation with no specific situated social act.
- Technical instruction, driving mechanics, self-defense, exercise, music,
  software, tools, cooking, or generic self-care.
- A solo person standing, gesturing, using a phone, eating, or journaling.
  Solo behavior counts only when the convention itself is visibly performed,
  such as greeting, dress etiquette, table manners, queuing, or playground and
  public-space conduct.

Use the whole time sequence. Do not infer a visual event from narration. Return
exactly one JSON object with:
visual_sequence: continuous_episode|roleplay_or_fiction|behavioral_tableau|solo_convention|presenter_or_interview|disconnected_broll_or_montage|technical_or_generic_activity|text_graphic_or_static_pose|fragmentary_or_none|unclear
participant_configuration: visible_interacting_participants|roleplayed_or_split_participants|visible_actor_and_visually_established_offscreen_party|recognizable_solo_convention|depicted_people_without_interaction|none|unclear
behavior_evidence: visibly_performed|visibly_depicted_tableau|only_spoken_or_inferred|none|unclear
specific_social_act: yes|no|unclear
presenter_or_program: yes|no|unclear
disconnected_broll: yes|no|unclear
technical_or_generic_activity: yes|no|unclear
literal_behavior: one short literal description, or none
evidence: one short sentence explaining what is and is not visibly established

Keep free text under 40 words. Return JSON only."""


ENUMS = {
    "visual_sequence": {
        "continuous_episode", "roleplay_or_fiction", "behavioral_tableau",
        "solo_convention", "presenter_or_interview",
        "disconnected_broll_or_montage", "technical_or_generic_activity",
        "text_graphic_or_static_pose", "fragmentary_or_none", "unclear",
    },
    "participant_configuration": {
        "visible_interacting_participants", "roleplayed_or_split_participants",
        "visible_actor_and_visually_established_offscreen_party",
        "recognizable_solo_convention", "depicted_people_without_interaction",
        "none", "unclear",
    },
    "behavior_evidence": {
        "visibly_performed", "visibly_depicted_tableau",
        "only_spoken_or_inferred", "none", "unclear",
    },
    "specific_social_act": {"yes", "no", "unclear"},
    "presenter_or_program": {"yes", "no", "unclear"},
    "disconnected_broll": {"yes", "no", "unclear"},
    "technical_or_generic_activity": {"yes", "no", "unclear"},
}
TEXT_FIELDS = {"literal_behavior", "evidence"}


def derive_demo_candidate(result: dict[str, Any]) -> bool:
    return (
        result["visual_sequence"] in {
            "continuous_episode", "roleplay_or_fiction", "behavioral_tableau",
            "solo_convention",
        }
        and result["participant_configuration"] in {
            "visible_interacting_participants", "roleplayed_or_split_participants",
            "visible_actor_and_visually_established_offscreen_party",
            "recognizable_solo_convention",
        }
        and result["behavior_evidence"] in {
            "visibly_performed", "visibly_depicted_tableau",
        }
        and result["specific_social_act"] == "yes"
        and result["presenter_or_program"] == "no"
        and result["disconnected_broll"] == "no"
        and result["technical_or_generic_activity"] == "no"
    )


def parse_demo_adjudicator(text: str) -> dict[str, Any]:
    result = _json_object(text)
    expected = set(ENUMS) | TEXT_FIELDS
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
        "max_tokens": 850,
        "messages": [
            {"role": "system", "content": SYSTEM_DEMO_ADJUDICATOR},
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
                **base_record(row, model),
                "rubric": "demo_adjudicator_v3",
                "quote_context": row["_quotes"],
                "result": parse_demo_adjudicator(content),
                "raw_response": content,
                "usage": body.get("usage"),
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            last_error = f"HTTPError {exc.code}: {exc.read().decode(errors='replace')[:2000]}"
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(2**attempt)
    return {
        **base_record(row, model),
        "rubric": "demo_adjudicator_v3",
        "quote_context": row["_quotes"],
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
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
            print(f"{index}/{len(futures)} {record['item_id']} error={record['error']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
