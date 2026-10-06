#!/usr/bin/env python3
"""Run a label-blind V22 visual-demonstration audit on dense storyboards."""

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
    from scripts.run_instructional_v15_structure_vlm import normalize_manifest_row
except ModuleNotFoundError:
    from run_open_vlm_v10_storyboards import (  # type: ignore[no-redef]
        _json_object,
        base_record,
        load_jsonl,
        sha256,
        successful_keys,
    )
    from run_instructional_v15_structure_vlm import normalize_manifest_row  # type: ignore[no-redef]


SYSTEM_V22 = """You are a LABEL-BLIND visual auditor of a dense chronological
video storyboard. You receive no title, transcript, norm, polarity, category,
or explanation. Judge what the pixels demonstrate, not what the video might be
about.

The desired item is a VISUAL DEMONSTRATION of socially interpretable behavior.
Either of these qualifies:
1. enacted_social_scenario: people or characters perform a concrete social act,
   exchange, conflict, response, helping behavior, etiquette behavior, or
   shared-space behavior in a connected episode;
2. explicit_behavior_demonstration: someone physically demonstrates a socially
   meaningful gesture, posture, signal, ritual, or do/don't behavior, even when
   only one demonstrator is visible.

Live action, role-play, film, animation, puppets, dolls, and screen-based acted
dialogue can qualify. Speech content need not be readable when the ordered
visual action itself establishes a connected social episode. A film or
animation is not disqualified merely because it is staged.

Do NOT pass:
- presenters, interviews, lectures, panels, speeches, court testimony, or
  ordinary people talking when no separate behavior is demonstrated;
- generic classroom participation, dining, standing in a group, device use, or
  other social activity without a concrete target behavior;
- news footage or documentary montage that merely illustrates a topic;
- stock footage, advertising, unrelated tutorials, slides, static graphics, or
  captions that state an action not visible in the pixels;
- aftermath, reaction, or allegation when the causative behavior is off-screen;
- random motion, scene cuts, multiple faces, or a social location by themselves.

Read frames left-to-right, top-to-bottom. Return exactly one JSON object with:
visual_role: enacted_social_scenario|explicit_behavior_demonstration|generic_social_activity|talking_head_or_interview|illustrative_broll_or_news|text_or_graphic|off_topic|uncertain
connected_episode_visible: yes|no|uncertain
concrete_performed_behavior_visible: yes|no|uncertain
socially_interpretable_from_pixels: yes|no|uncertain
only_explanation_report_or_broll: yes|no|uncertain
literal_visible_behavior: short literal description or none
demo_pass: yes|no|uncertain
evidence: one short sentence

demo_pass=yes only when visual_role is enacted_social_scenario or
explicit_behavior_demonstration, concrete_performed_behavior_visible=yes,
socially_interpretable_from_pixels=yes, and
only_explanation_report_or_broll=no. Otherwise fail closed."""


TRI = {"yes", "no", "uncertain"}
ROLES = {
    "enacted_social_scenario",
    "explicit_behavior_demonstration",
    "generic_social_activity",
    "talking_head_or_interview",
    "illustrative_broll_or_news",
    "text_or_graphic",
    "off_topic",
    "uncertain",
}
PASS_ROLES = {"enacted_social_scenario", "explicit_behavior_demonstration"}


def parse_v22(text: str) -> dict[str, Any]:
    result = _json_object(text)
    expected = {
        "visual_role",
        "connected_episode_visible",
        "concrete_performed_behavior_visible",
        "socially_interpretable_from_pixels",
        "only_explanation_report_or_broll",
        "literal_visible_behavior",
        "demo_pass",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    if result["visual_role"] not in ROLES:
        raise ValueError("invalid visual_role")
    for key in (
        "connected_episode_visible",
        "concrete_performed_behavior_visible",
        "socially_interpretable_from_pixels",
        "only_explanation_report_or_broll",
        "demo_pass",
    ):
        if result[key] not in TRI:
            raise ValueError(f"invalid {key}")
    for key in ("literal_visible_behavior", "evidence"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"invalid {key}")
    strict = (
        result["visual_role"] in PASS_ROLES
        and result["concrete_performed_behavior_visible"] == "yes"
        and result["socially_interpretable_from_pixels"] == "yes"
        and result["only_explanation_report_or_broll"] == "no"
    )
    if result["demo_pass"] == "yes" and not strict:
        result["demo_pass_raw"] = "yes"
        result["demo_pass"] = "uncertain"
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
        raise ValueError(f"storyboard hash mismatch: {row['item_id']}")
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 500,
        "messages": [
            {"role": "system", "content": SYSTEM_V22},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": sheet.resolve().as_uri()}},
                    {"type": "text", "text": "Audit the complete ordered storyboard."},
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
                **base_record(row, model),
                "rubric": "v22_label_blind_action_demo",
                "result": parse_v22(content),
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
        "rubric": "v22_label_blind_action_demo",
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def rebase_sheet_paths(
    rows: list[dict[str, Any]], sheet_root: Path | None
) -> list[dict[str, Any]]:
    if sheet_root is None:
        return rows
    return [
        {**row, "sheet_path": str(sheet_root / Path(row["sheet_path"]).name)}
        for row in rows
    ]


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
        "--sheet-root",
        type=Path,
        help="Rebase storyboard basenames onto a host-local directory",
    )
    args = parser.parse_args()
    rows = rebase_sheet_paths([
        normalize_manifest_row(row, args.manifest.parent, "instructional")
        for row in load_jsonl(args.manifest)
    ], args.sheet_root)
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
