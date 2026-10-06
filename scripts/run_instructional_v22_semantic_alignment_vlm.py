#!/usr/bin/env python3
"""Audit whether a visible instructional demonstration matches its weak label."""

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


SYSTEM_ALIGNMENT = """You audit whether a visible instructional demonstration
actually grounds its proposed weak label. You receive a dense chronological
storyboard plus a proposed norm, polarity, explanation, and transcript bounds.

First describe only the concrete behavior visible in the pixels. Then compare
that behavior with the proposed label. Do not infer an action merely because the
transcript, norm, or explanation names it. Animation, role-play, puppets, film,
screen-based acted dialogue, and a one-person physical behavior demonstration
are valid media. Talking heads, slides, topical b-roll, generic social activity,
and reports of an off-screen action are not demonstrations.

For violation or correct polarity, the depicted behavior must support that
polarity. For explanation polarity, polarity_alignment may be not_applicable,
but a concrete behavior matching the claimed norm must still be demonstrated.
A complete demonstration contains enough of the behavior to learn what was
done; setup-only, aftermath-only, or a tiny ambiguous fragment is partial.

Return exactly one JSON object with:
observed_visible_behavior: short literal description or none
claimed_behavior: short description derived from the supplied weak label
behavior_alignment: yes|partial|no|uncertain
polarity_alignment: yes|no|not_applicable|uncertain
demonstration_completeness: complete|partial|none|uncertain
label_grounded_in_pixels: yes|no|uncertain
semantic_pass: yes|no|uncertain
evidence: one short sentence

semantic_pass=yes only when behavior_alignment=yes,
demonstration_completeness=complete, label_grounded_in_pixels=yes, and
polarity_alignment is yes or legitimately not_applicable for explanation
polarity. Otherwise fail closed."""

SYSTEM_SITUATED_ALIGNMENT = """You audit whether an instructional clip grounds
its proposed social-norm weak label. You receive a dense chronological
storyboard plus the norm, polarity, explanation, and transcript bounds from the
same saved interval.

Two kinds of demonstrated behavior qualify:
1. physical: the target action and its participant, affected person, or shared
   public setting are visible in a connected episode;
2. situated spoken act: the supplied interval quote enacts the target behavior
   while the storyboard visibly shows the speaker interacting with the affected
   participant or a genuine shared-setting audience in that same episode.

For situated speech, the interval transcript is evidence of what is audibly
said; do not require words to be readable in pixels. But fail direct-to-camera
presenters, interviews, lectures, panels, advice, sample phrases, generic
conversation, and reports about an off-screen event. Also fail stock or news
b-roll, slides, solo technical/professional performance, safety or medical
procedures, and physical activity whose social meaning comes only from the
metadata. The behavior must govern another person's treatment, rights, welfare,
or expectations, or conduct in a genuinely shared public setting.

Check polarity strictly. A violation must enact the prohibited behavior. A
correct example must sincerely enact the desired behavior; sarcasm, mockery, a
hostile continuation, or an incompatible response invalidates it. A contrast
must show both sides or an explicit correction. An explanation still needs an
actual paired demonstration. A complete demonstration includes the causative
act or situated utterance and enough participant response/context to identify
the event, rather than setup, aftermath, or an ambiguous fragment.

Return exactly one JSON object with:
observed_visible_behavior: short literal description or none
claimed_behavior: short description derived from the supplied weak label
behavior_alignment: yes|partial|no|uncertain
polarity_alignment: yes|no|not_applicable|uncertain
demonstration_completeness: complete|partial|none|uncertain
label_grounded_in_pixels: yes|no|uncertain
semantic_pass: yes|no|uncertain
evidence: one short sentence

For this rubric, label_grounded_in_pixels=yes means either a physical act is
visible or a situated spoken act is anchored to visibly present same-episode
participants as defined above. semantic_pass=yes only when behavior_alignment
is yes, demonstration_completeness is complete, label_grounded_in_pixels is
yes, and polarity_alignment is yes (or legitimately not_applicable for an
explanation). Otherwise fail closed. Keep every string field under 35 words."""


ALIGNMENT = {"yes", "partial", "no", "uncertain"}
POLARITY = {"yes", "no", "not_applicable", "uncertain"}
COMPLETENESS = {"complete", "partial", "none", "uncertain"}
TRI = {"yes", "no", "uncertain"}


def parse_alignment(text: str, proposed_polarity: str) -> dict[str, Any]:
    result = _json_object(text)
    expected = {
        "observed_visible_behavior",
        "claimed_behavior",
        "behavior_alignment",
        "polarity_alignment",
        "demonstration_completeness",
        "label_grounded_in_pixels",
        "semantic_pass",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    for key in ("observed_visible_behavior", "claimed_behavior", "evidence"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"invalid {key}")
    if result["behavior_alignment"] not in ALIGNMENT:
        raise ValueError("invalid behavior_alignment")
    if result["polarity_alignment"] not in POLARITY:
        raise ValueError("invalid polarity_alignment")
    if result["demonstration_completeness"] not in COMPLETENESS:
        raise ValueError("invalid demonstration_completeness")
    for key in ("label_grounded_in_pixels", "semantic_pass"):
        if result[key] not in TRI:
            raise ValueError(f"invalid {key}")
    polarity_ok = result["polarity_alignment"] == "yes" or (
        proposed_polarity == "explanation"
        and result["polarity_alignment"] == "not_applicable"
    )
    strict = (
        result["behavior_alignment"] == "yes"
        and result["demonstration_completeness"] == "complete"
        and result["label_grounded_in_pixels"] == "yes"
        and polarity_ok
    )
    if result["semantic_pass"] == "yes" and not strict:
        result["semantic_pass_raw"] = "yes"
        result["semantic_pass"] = "uncertain"
        result["consistency_repair"] = "inconsistent_positive_to_uncertain"
    return result


def semantic_context(row: dict[str, Any]) -> dict[str, str]:
    return {
        "norm": str(row.get("norm") or ""),
        "polarity": str(row.get("polarity") or ""),
        "explanation": str(row.get("explanation") or ""),
        "start_quote": str(row.get("start_quote") or ""),
        "end_quote": str(row.get("end_quote") or ""),
    }


def join_manifests(
    storyboard_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    semantics = {str(row["item_id"]): row for row in semantic_rows}
    if len(semantics) != len(semantic_rows):
        raise ValueError("duplicate semantic item_id")
    joined = []
    for row in storyboard_rows:
        item_id = str(row["item_id"])
        if item_id not in semantics:
            raise ValueError(f"missing semantic row: {item_id}")
        joined.append({**row, "_semantic": semantic_context(semantics[item_id])})
    return joined


def request_one(
    endpoint: str,
    model: str,
    row: dict[str, Any],
    timeout: int,
    retries: int,
    rubric: str = "v22_pixels",
) -> dict[str, Any]:
    sheet = Path(row["sheet_path"])
    if not sheet.is_absolute():
        sheet = Path(row["_manifest_dir"]) / sheet
    if sha256(sheet) != row["sheet_sha256"]:
        raise ValueError(f"storyboard hash mismatch: {row['item_id']}")
    context = row["_semantic"]
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 650,
        "messages": [
            {
                "role": "system",
                "content": (
                    SYSTEM_SITUATED_ALIGNMENT
                    if rubric == "v23_situated_speech"
                    else SYSTEM_ALIGNMENT
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": sheet.resolve().as_uri()}},
                    {
                        "type": "text",
                        "text": "Proposed weak label:\n" + json.dumps(context, sort_keys=True),
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
                **base_record(row, model),
                "rubric": rubric,
                "semantic_context": context,
                "result": parse_alignment(content, context["polarity"]),
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
        "rubric": rubric,
        "semantic_context": context,
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
    parser.add_argument(
        "--rubric",
        choices=("v22_pixels", "v23_situated_speech"),
        default="v22_pixels",
    )
    parser.add_argument("--sheet-root", type=Path)
    args = parser.parse_args()
    storyboard_rows = [
        normalize_manifest_row(row, args.manifest.parent, "instructional")
        for row in load_jsonl(args.manifest)
    ]
    if args.sheet_root is not None:
        storyboard_rows = [
            {**row, "sheet_path": str(args.sheet_root / Path(row["sheet_path"]).name)}
            for row in storyboard_rows
        ]
    rows = join_manifests(storyboard_rows, load_jsonl(args.semantic_manifest))
    completed = successful_keys(load_jsonl(args.out)) if args.out.exists() else set()
    pending = [row for row in rows if (row["item_id"], args.model) not in completed]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                request_one,
                args.endpoint,
                args.model,
                row,
                args.timeout,
                args.retries,
                args.rubric,
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
