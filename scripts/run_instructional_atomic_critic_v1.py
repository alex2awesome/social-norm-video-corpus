#!/usr/bin/env python3
"""Run an operational, decomposed instructional-demo critic over storyboards."""

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
    from scripts.run_instructional_v22_semantic_alignment_vlm import (
        join_manifests,
    )
except ModuleNotFoundError:
    from run_open_vlm_v10_storyboards import (  # type: ignore[no-redef]
        _json_object,
        base_record,
        load_jsonl,
        sha256,
        successful_keys,
    )
    from run_instructional_v15_structure_vlm import normalize_manifest_row  # type: ignore[no-redef]
    from run_instructional_v22_semantic_alignment_vlm import join_manifests  # type: ignore[no-redef]


SYSTEM_ATOMIC = """You audit an instructional clip as a possible visual
demonstration of a proposed social-norm label. You receive a dense chronological
storyboard and transcript evidence from that exact saved interval. Do not make
one vague "is this a social norm?" judgment. Fill the operational fields below.

First state the concrete actor-action-target episode. A valid episode may be
live action, role-play, film/TV, animation, puppets, split-screen dialogue, or
one actor playing multiple roles. Situated speech counts when the supplied
quote is spoken to an affected participant or genuine same-episode audience
shown in the storyboard. Presenters, interviews, panels, advice, reports about
off-screen events, and narration over generic b-roll are not connected episodes.

Choose social_basis from concrete effect types. interpersonal_treatment means
how one person treats another; relationship_duty means conduct owed within a
relationship; institutional_fairness means treatment by a school, workplace,
business, or authority; public_shared_conduct means conduct affecting a public,
community, or shared environment; care_or_welfare means protecting another
person's rights, autonomy, safety, or welfare. self_only and
technical_procedure are not social obligations for this dataset.

Compare the observed act to the proposed rule independently of the supplied
polarity. relation=complies means the act follows the rule; violates means it
breaks the rule; contrast requires both sides or an explicit correction;
illustrates is only for a relevant demonstrated episode whose direction truly
cannot be reduced to complies/violates. Do not call a correct act a violation
merely because the metadata says violation.

Return exactly one JSON object with these fields:
episode_medium: live_action|roleplay|film_tv|animation|puppets|screen_dialogue|other|none|unclear
actor_action: short literal actor-action description, or none
affected_party: short literal target/affected party, or none
grounding: visible_physical|situated_speech_to_present_party|both|metadata_or_narration_only|none|unclear
episode_continuity: connected|fragmentary|presenter_or_interview|report_or_broll|none|unclear
social_basis: interpersonal_treatment|relationship_duty|institutional_fairness|public_shared_conduct|care_or_welfare|self_only|technical_procedure|none|unclear
relation: complies|violates|contrast|illustrates|unrelated|unclear
completeness: complete|partial|none|unclear
evidence: one short sentence tying the choices to the interval

Use complete only when the interval contains enough of the causative act or
situated utterance and context to learn what happened. Keep each free-text field
under 35 words. Return JSON only."""


ENUMS = {
    "episode_medium": {
        "live_action", "roleplay", "film_tv", "animation", "puppets",
        "screen_dialogue", "other", "none", "unclear",
    },
    "grounding": {
        "visible_physical", "situated_speech_to_present_party", "both",
        "metadata_or_narration_only", "none", "unclear",
    },
    "episode_continuity": {
        "connected", "fragmentary", "presenter_or_interview",
        "report_or_broll", "none", "unclear",
    },
    "social_basis": {
        "interpersonal_treatment", "relationship_duty",
        "institutional_fairness", "public_shared_conduct", "care_or_welfare",
        "self_only", "technical_procedure", "none", "unclear",
    },
    "relation": {
        "complies", "violates", "contrast", "illustrates", "unrelated",
        "unclear",
    },
    "completeness": {"complete", "partial", "none", "unclear"},
}
TEXT_FIELDS = {"actor_action", "affected_party", "evidence"}
SOCIAL_BASES = {
    "interpersonal_treatment", "relationship_duty", "institutional_fairness",
    "public_shared_conduct", "care_or_welfare",
}
GROUNDED = {"visible_physical", "situated_speech_to_present_party", "both"}


def relation_matches_polarity(relation: str, polarity: str) -> bool:
    polarity = polarity.strip().lower()
    if polarity == "violation":
        return relation == "violates"
    if polarity == "correct":
        return relation == "complies"
    if polarity == "contrast":
        return relation == "contrast"
    if polarity == "explanation":
        return relation in {"complies", "violates", "contrast", "illustrates"}
    return False


def derive_atomic_pass(result: dict[str, Any], polarity: str) -> bool:
    return (
        result["grounding"] in GROUNDED
        and result["episode_continuity"] == "connected"
        and result["social_basis"] in SOCIAL_BASES
        and result["completeness"] == "complete"
        and relation_matches_polarity(result["relation"], polarity)
    )


def parse_atomic(text: str, polarity: str) -> dict[str, Any]:
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
    result["relation_matches_proposed_polarity"] = relation_matches_polarity(
        result["relation"], polarity
    )
    result["atomic_pass"] = derive_atomic_pass(result, polarity)
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
    context = row["_semantic"]
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 750,
        "messages": [
            {"role": "system", "content": SYSTEM_ATOMIC},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": sheet.resolve().as_uri()}},
                    {
                        "type": "text",
                        "text": "Proposed weak label and interval transcript:\n"
                        + json.dumps(context, sort_keys=True),
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
                "rubric": "atomic_v1",
                "semantic_context": context,
                "result": parse_atomic(content, context["polarity"]),
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
        "rubric": "atomic_v1",
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
                request_one, args.endpoint, args.model, row, args.timeout, args.retries
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
