#!/usr/bin/env python3
"""Extract a label-blind instructional episode using pixels and interval quotes."""

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
    from scripts.run_instructional_v15_structure_vlm import normalize_manifest_row
except ModuleNotFoundError:
    from run_open_vlm_v10_storyboards import (  # type: ignore[no-redef]
        _json_object, base_record, load_jsonl, sha256, successful_keys,
    )
    from run_instructional_v15_structure_vlm import normalize_manifest_row  # type: ignore[no-redef]


SYSTEM_BLIND_EPISODE = """You extract literal evidence from an instructional
clip without seeing its title, category, proposed norm, polarity, or
explanation. You receive a dense chronological storyboard and the start/end
transcript quotes from the same saved interval. Do not guess what lesson or
social norm the source intended.

Identify only the episode that actually occurs. Live action, film/TV,
animation, puppets, split-screen dialogue, and one actor playing multiple roles
can form connected or role-play episodes. A supplied quote is a situated spoken
act only if the storyboard shows it addressed to an affected participant or a
genuine same-episode audience. Direct-to-camera presentation, interviews,
panels, advice, sample phrases, and reports about off-screen events are not
situated acts. Narration does not make generic b-roll an episode.

physical_social_action=yes only when the pixels show a concrete action directed
at another participant or affecting people in a shared setting. Object
manipulation, sport, driving, self-defense, food/safety/medical mechanics,
software use, and generic conversation are no unless a distinct social act is
actually performed. Do not infer an affected person from the transcript alone.

Return exactly one JSON object with:
episode_medium: live_action|roleplay|film_tv|animation|puppets|screen_dialogue|technical_or_solo_demo|other|none|unclear
visual_context: connected_episode|roleplay_episode|presenter_or_interview|report_or_broll|technical_or_solo_demo|text_or_graphic|fragmentary|none|unclear
visible_participants: short literal description, or none
literal_visible_action: short literal action, or none
quote_speech_act: short literal speech act from the supplied quote, or none
quote_function: situated_to_present_party|narration_or_report|presenter_advice|sample_or_hypothetical|none_or_unavailable|unclear
participant_grounding: affected_party_present|shared_audience_present|actor_only|none|unclear
physical_social_action: yes|no|unclear
episode_complete: yes|no|unclear
evidence: one short sentence

Keep free text under 35 words. Return JSON only."""


ENUMS = {
    "episode_medium": {
        "live_action", "roleplay", "film_tv", "animation", "puppets",
        "screen_dialogue", "technical_or_solo_demo", "other", "none", "unclear",
    },
    "visual_context": {
        "connected_episode", "roleplay_episode", "presenter_or_interview",
        "report_or_broll", "technical_or_solo_demo", "text_or_graphic",
        "fragmentary", "none", "unclear",
    },
    "quote_function": {
        "situated_to_present_party", "narration_or_report", "presenter_advice",
        "sample_or_hypothetical", "none_or_unavailable", "unclear",
    },
    "participant_grounding": {
        "affected_party_present", "shared_audience_present", "actor_only",
        "none", "unclear",
    },
    "physical_social_action": {"yes", "no", "unclear"},
    "episode_complete": {"yes", "no", "unclear"},
}
TEXT_FIELDS = {
    "visible_participants", "literal_visible_action", "quote_speech_act", "evidence"
}


def derive_episode_pass(result: dict[str, Any]) -> bool:
    episode = result["visual_context"] in {"connected_episode", "roleplay_episode"}
    participant = result["participant_grounding"] in {
        "affected_party_present", "shared_audience_present"
    }
    behavior = (
        result["physical_social_action"] == "yes"
        or result["quote_function"] == "situated_to_present_party"
    )
    return episode and participant and behavior and result["episode_complete"] == "yes"


def parse_blind_episode(text: str) -> dict[str, Any]:
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
    result["episode_pass"] = derive_episode_pass(result)
    return result


def quotes_by_item(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    output = {}
    for row in rows:
        item_id = str(row["item_id"])
        if item_id in output:
            raise ValueError(f"duplicate semantic item_id: {item_id}")
        output[item_id] = {
            "start_quote": str(row.get("start_quote") or ""),
            "end_quote": str(row.get("end_quote") or ""),
        }
    return output


def request_one(endpoint: str, model: str, row: dict[str, Any], timeout: int, retries: int) -> dict[str, Any]:
    sheet = Path(row["sheet_path"])
    if not sheet.is_absolute():
        sheet = Path(row["_manifest_dir"]) / sheet
    if sha256(sheet) != row["sheet_sha256"]:
        raise ValueError(f"storyboard hash mismatch: {row['item_id']}")
    quotes = row["_quotes"]
    payload = {
        "model": model, "temperature": 0, "max_tokens": 750,
        "messages": [
            {"role": "system", "content": SYSTEM_BLIND_EPISODE},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": sheet.resolve().as_uri()}},
                {"type": "text", "text": "Interval quotes only:\n" + json.dumps(quotes, sort_keys=True)},
            ]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode(); last_error = ""; last_content = None
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
                **base_record(row, model), "rubric": "blind_episode_v1",
                "quote_context": quotes, "result": parse_blind_episode(content),
                "raw_response": content, "usage": body.get("usage"), "error": None,
            }
        except urllib.error.HTTPError as exc:
            last_error = f"HTTPError {exc.code}: {exc.read().decode(errors='replace')[:2000]}"
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(2**attempt)
    return {
        **base_record(row, model), "rubric": "blind_episode_v1",
        "quote_context": quotes, "result": None, "raw_response": last_content,
        "usage": None, "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=2); parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2); parser.add_argument("--sheet-root", type=Path)
    args = parser.parse_args()
    rows = [normalize_manifest_row(row, args.manifest.parent, "instructional") for row in load_jsonl(args.manifest)]
    if args.sheet_root is not None:
        rows = [{**row, "sheet_path": str(args.sheet_root / Path(row["sheet_path"]).name)} for row in rows]
    quotes = quotes_by_item(load_jsonl(args.semantic_manifest))
    if {str(row["item_id"]) for row in rows} != set(quotes):
        raise ValueError("storyboard and semantic cohorts differ")
    rows = [{**row, "_quotes": quotes[str(row["item_id"])]} for row in rows]
    completed = successful_keys(load_jsonl(args.out)) if args.out.exists() else set()
    pending = [row for row in rows if (row["item_id"], args.model) not in completed]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(request_one, args.endpoint, args.model, row, args.timeout, args.retries): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result(); handle.write(json.dumps(record, sort_keys=True) + "\n"); handle.flush()
            print(f"{index}/{len(futures)} {record['item_id']} error={record['error']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
