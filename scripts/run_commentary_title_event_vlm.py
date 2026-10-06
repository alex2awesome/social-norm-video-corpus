#!/usr/bin/env python3
"""Run a title-conditioned event-retrieval VLM over commentary storyboards.

This is intentionally a permissive localization pass, not an acceptance rule.
It asks whether any part of the video depicts the title-named event and emits
candidate bounds for human motion/crop review.
"""

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


SYSTEM = """You localize possible title-named events in complete video
storyboards. The title is fallible and may overclaim. Do not trust it as
evidence. Search every panel, including tiny news insets and brief sequences;
ignore how much of the video is presenters, interviews, graphics, or b-roll.

Your job is retrieval, not moral judgment. A candidate exists when ordered
frames might show the title-named actor performing the title-named action
toward the named target or property. Physical interaction, property taking,
dangerous driving, and a situated verbal act can qualify. Aftermath, an
interview, captions asserting an off-screen act, ordinary object handling, and
mere proximity do not establish the action.

Read frames left-to-right, then top-to-bottom. Printed timestamps are seconds.
If stills are insufficient but a bounded moving clip, crop, or audio could
resolve the event, preserve it as uncertain and give the narrowest useful
candidate window. Do not fail a source merely because most panels are
presenters. Do not mark yes unless the action itself is visibly established.

Return one JSON object with exactly:
candidate_event: yes|no|uncertain
title_action_visible: yes|no|uncertain
title_actor_visible: yes|no|uncertain
title_target_or_property_visible: yes|no|uncertain
actor_action_target_same_event: yes|no|uncertain
event_kind: physical_interaction|property_taking|dangerous_driving|situated_verbal_act|other_social_event|non_social_or_absent|uncertain
candidate_start_sec: number|null
candidate_end_sec: number|null
followup: none|motion|audio|crop|motion_and_audio|motion_and_crop|unresolvable
literal_candidate: short literal description
title_alignment: exact|same_norm_different_details|mismatch|unresolved|no_event
evidence: one short sentence citing visible frames/timestamps or the missing action

candidate_event=yes requires title_action_visible=yes and
actor_action_target_same_event=yes. Use candidate_event=uncertain whenever
motion/audio/crop is still required. Bounds are required for yes or uncertain,
must be ordered, and must be null for no."""


TRI = {"yes", "no", "uncertain"}
EVENT_KIND = {
    "physical_interaction",
    "property_taking",
    "dangerous_driving",
    "situated_verbal_act",
    "other_social_event",
    "non_social_or_absent",
    "uncertain",
}
FOLLOWUP = {
    "none",
    "motion",
    "audio",
    "crop",
    "motion_and_audio",
    "motion_and_crop",
    "unresolvable",
}
ALIGNMENT = {
    "exact",
    "same_norm_different_details",
    "mismatch",
    "unresolved",
    "no_event",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def join_manifests(
    visual_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join render records to sealed semantics across old and current schemas.

    Older title-event runs put ``candidate_id`` on both records.  The generic
    storyboard renderer used by the full-corpus expansion intentionally emits
    only ``item_id``.  Prefer candidate_id when both sides have it, otherwise
    use item_id.  Render failures remain in the manifest but are not submitted
    as image requests because they have no verifiable sheet.
    """
    by_candidate: dict[str, dict[str, Any]] = {}
    by_item: dict[str, dict[str, Any]] = {}
    for row in semantic_rows:
        candidate_id = row.get("candidate_id")
        item_id = row.get("item_id")
        if candidate_id:
            if candidate_id in by_candidate:
                raise ValueError("duplicate semantic candidate_id")
            by_candidate[str(candidate_id)] = row
        if item_id:
            if item_id in by_item:
                raise ValueError("duplicate semantic item_id")
            by_item[str(item_id)] = row
    result = []
    for visual in visual_rows:
        candidate_id = visual.get("candidate_id")
        item_id = visual.get("item_id")
        source = (
            by_candidate.get(str(candidate_id)) if candidate_id else None
        ) or (by_item.get(str(item_id)) if item_id else None)
        lookup = candidate_id or item_id
        if source is None:
            raise ValueError(f"missing semantic row for {lookup}")
        row = dict(visual)
        row["candidate_id"] = str(
            candidate_id or source.get("candidate_id") or item_id
        )
        if "audit_index" not in row:
            audit_index = source.get("audit_index", visual.get("storyboard_index"))
            if not isinstance(audit_index, int):
                raise ValueError(f"missing integer audit_index for {lookup}")
            row["audit_index"] = audit_index
        row["title"] = source["norm"]
        result.append(row)
    return result


def resolve_sheet(
    row: dict[str, Any], manifest_dir: Path, sheet_root: Path | None
) -> Path | None:
    """Resolve a sealed sheet locally, optionally rebasing an absolute path."""
    raw = row.get("sheet_path")
    if not raw:
        return None
    sheet = Path(str(raw))
    if sheet_root is not None:
        # Audit packages copy rendered sheets under one root.  Only the sealed
        # basename is rebased; the SHA-256 below still enforces image identity.
        sheet = sheet_root / sheet.name
    elif not sheet.is_absolute():
        sheet = manifest_dir / sheet
    return sheet


def json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def parse_result(text: str) -> dict[str, Any]:
    result = json_object(text)
    expected = {
        "candidate_event",
        "title_action_visible",
        "title_actor_visible",
        "title_target_or_property_visible",
        "actor_action_target_same_event",
        "event_kind",
        "candidate_start_sec",
        "candidate_end_sec",
        "followup",
        "literal_candidate",
        "title_alignment",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    tri_fields = {
        "candidate_event",
        "title_action_visible",
        "title_actor_visible",
        "title_target_or_property_visible",
        "actor_action_target_same_event",
    }
    for key in tri_fields:
        if result[key] not in TRI:
            raise ValueError(f"invalid {key}: {result[key]!r}")
    if result["event_kind"] not in EVENT_KIND:
        raise ValueError(f"invalid event_kind: {result['event_kind']!r}")
    if result["followup"] not in FOLLOWUP:
        raise ValueError(f"invalid followup: {result['followup']!r}")
    if result["title_alignment"] not in ALIGNMENT:
        raise ValueError(f"invalid title_alignment: {result['title_alignment']!r}")
    for key in ("literal_candidate", "evidence"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{key} must be a non-empty string")

    start = result["candidate_start_sec"]
    end = result["candidate_end_sec"]
    if result["candidate_event"] == "no":
        if start is not None or end is not None:
            result["candidate_start_sec"] = None
            result["candidate_end_sec"] = None
            result["bounds_repair"] = "no_event_bounds_cleared"
    elif (
        not isinstance(start, (int, float))
        or isinstance(start, bool)
        or not isinstance(end, (int, float))
        or isinstance(end, bool)
        or float(start) < 0
        or float(end) <= float(start)
    ):
        result["candidate_event_raw"] = result["candidate_event"]
        result["candidate_event"] = "uncertain"
        result["candidate_start_sec"] = None
        result["candidate_end_sec"] = None
        result["bounds_repair"] = "invalid_bounds_to_uncertain"
    if result["candidate_event"] == "yes" and not (
        result["title_action_visible"] == "yes"
        and result["actor_action_target_same_event"] == "yes"
    ):
        result["candidate_event_raw"] = "yes"
        result["candidate_event"] = "uncertain"
        result["consistency_repair"] = "unsupported_yes_to_uncertain"
    return result


def request_one(
    endpoint: str,
    model: str,
    row: dict[str, Any],
    manifest_dir: Path,
    sheet_root: Path | None,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    sheet = resolve_sheet(row, manifest_dir, sheet_root)
    if sheet is None:
        raise ValueError(f"no rendered sheet: {row['candidate_id']}")
    if sha256(sheet) != row["sheet_sha256"]:
        raise ValueError(f"sheet hash mismatch: {row['candidate_id']}")
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 600,
        "messages": [
            {"role": "system", "content": SYSTEM},
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
                            f"Fallible source title: {row['title']}\n"
                            "Search the complete storyboard and localize any "
                            "possible matching event."
                        ),
                    },
                ],
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode()
    last_error = ""
    last_content: str | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            last_content = body["choices"][0]["message"].get("content")
            if not last_content:
                raise ValueError("empty response")
            return {
                "audit_index": row["audit_index"],
                "candidate_id": row["candidate_id"],
                "item_id": row["item_id"],
                "model": model,
                "rubric": "commentary_title_event_retrieval_v1",
                "title": row["title"],
                "sheet_sha256": row["sheet_sha256"],
                "result": parse_result(last_content),
                "raw_response": last_content,
                "usage": body.get("usage"),
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
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2**attempt)
    return {
        "audit_index": row["audit_index"],
        "candidate_id": row["candidate_id"],
        "item_id": row["item_id"],
        "model": model,
        "rubric": "commentary_title_event_retrieval_v1",
        "title": row["title"],
        "sheet_sha256": row["sheet_sha256"],
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-manifest", type=Path, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--sheet-root",
        type=Path,
        help="Directory containing copied sheets; sealed hashes are verified.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = join_manifests(
        load_jsonl(args.visual_manifest),
        load_jsonl(args.semantic_manifest),
    )
    completed = {
        (row["candidate_id"], row["model"])
        for row in load_jsonl(args.out)
        if row.get("result") is not None and row.get("error") is None
    } if args.out.exists() else set()
    pending = [
        row for row in rows
        if (row["candidate_id"], args.model) not in completed
        and row.get("sheet_path")
        and row.get("sheet_sha256")
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    produced: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                request_one,
                args.endpoint,
                args.model,
                row,
                args.visual_manifest.parent,
                args.sheet_root,
                args.timeout,
                args.retries,
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
                    "item_id": row["item_id"],
                    "model": args.model,
                    "rubric": "commentary_title_event_retrieval_v1",
                    "title": row["title"],
                    "sheet_sha256": row["sheet_sha256"],
                    "result": None,
                    "raw_response": None,
                    "usage": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if count % 10 == 0 or count == len(futures):
                print(f"{count}/{len(futures)} requests completed", flush=True)

    existing = load_jsonl(args.out) if args.out.exists() else []
    combined = existing + [produced[index] for index in sorted(produced)]
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined)
    )
    failures = sum(row.get("error") is not None for row in produced.values())
    print(
        json.dumps(
            {
                "requested": len(pending),
                "completed": len(produced),
                "failures": failures,
                "model": args.model,
            },
            sort_keys=True,
        )
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
