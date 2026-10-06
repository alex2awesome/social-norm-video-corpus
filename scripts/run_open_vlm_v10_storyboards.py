#!/usr/bin/env python3
"""Run a label-blind V10 social-demo audit over dense temporal storyboards."""

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


SYSTEM_V10A = """You are a LABEL-BLIND auditor of video storyboards. Decide
whether the saved clip actually PERFORMS a socially evaluable behavior. You are
not given a title, transcript, category, proposed norm, polarity, or
explanation. Never infer them.

A usable demo can be live action, role-play, film/TV, animation, puppets, a
screen scenario, static illustrated dialogue, or an explicit etiquette
demonstration. It needs an actor/character, the performed physical or situated
speech behavior, and its affected person or genuinely shared public setting in
the same episode. A later reaction is helpful but not required.

Fail these recurrent look-alikes:
- a presenter/interviewee says example words to the audience or describes what
  somebody did, without an affected person inside the depicted situation;
- generic conversation, gesturing, handshaking, eating, driving, transferring,
  cleaning, playing, or object handling whose social meaning is not visible;
- a formal safety, traffic, legal, medical/service procedure, house rule,
  developmental exercise, diagnostic task, motor skill, or technical lesson;
- B-roll, montage, aftermath, portraits, reaction faces, or text asserting an
  off-screen act;
- ordinary help or service activity that metadata could later moralize.

Do not reject an explicit social custom merely because it is pedagogical: a
performed bow, greeting, handshake, turn-taking exchange, consent dialogue, or
table-etiquette contrast can qualify. Direct-to-camera speech qualifies only
when it is the actual targeted social act in a depicted interaction, not advice
or a sample phrase.

The image is a temporal storyboard. Read frames left-to-right, then
top-to-bottom. Record literal evidence only.

Return one JSON object with exactly:
performed_social_behavior: yes|no|uncertain
actor_action_target_same_event: yes|no|uncertain
target_behavior_performed_not_described: yes|no|uncertain
affected_party_or_shared_setting_present: yes|no|uncertain
socially_evaluable_without_metadata: yes|no|uncertain
social_scope: tacit_interpersonal|shared_public|explicit_social_etiquette|formal_safety_or_legal|technical_medical_or_service_procedure|developmental_or_clinical_exercise|non_social_or_ordinary_action|uncertain
scene_role: situated_scene|roleplay_demo|animation_or_story_demo|text_dialogue_demo|presenter_sample_or_advice|retrospective_interview|generic_broll_or_montage|procedure_or_task_only|uncertain
social_response_or_consequence_present: yes|no|uncertain
literal_actor: short literal description or none
literal_action_or_situated_utterance: short literal description or none
literal_affected_party_or_shared_setting: short literal description or none
demo_usable: yes|no|uncertain
rejection_reason: none|behavior_only_described|no_affected_party|metadata_supplies_social_meaning|formal_or_procedural_only|developmental_or_non_social_task|broll_or_montage|uncertain
evidence: one short sentence naming the literal performed event or missing requirement

demo_usable=yes only when the first five fields are yes, social_scope is
tacit_interpersonal, shared_public, or explicit_social_etiquette, and scene_role
is situated_scene, roleplay_demo, animation_or_story_demo, or
text_dialogue_demo. Otherwise fail closed."""


YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}
SOCIAL_SCOPES = {
    "tacit_interpersonal",
    "shared_public",
    "explicit_social_etiquette",
    "formal_safety_or_legal",
    "technical_medical_or_service_procedure",
    "developmental_or_clinical_exercise",
    "non_social_or_ordinary_action",
    "uncertain",
}
SCENE_ROLES = {
    "situated_scene",
    "roleplay_demo",
    "animation_or_story_demo",
    "text_dialogue_demo",
    "presenter_sample_or_advice",
    "retrospective_interview",
    "generic_broll_or_montage",
    "procedure_or_task_only",
    "uncertain",
}
REJECTION_REASONS = {
    "none",
    "behavior_only_described",
    "no_affected_party",
    "metadata_supplies_social_meaning",
    "formal_or_procedural_only",
    "developmental_or_non_social_task",
    "broll_or_montage",
    "uncertain",
}
QUALIFYING_SCOPES = {
    "tacit_interpersonal",
    "shared_public",
    "explicit_social_etiquette",
}
QUALIFYING_ROLES = {
    "situated_scene",
    "roleplay_demo",
    "animation_or_story_demo",
    "text_dialogue_demo",
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(text: str) -> dict[str, Any]:
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
        raise ValueError("expected one JSON object")
    return value


def parse_v10a(text: str) -> dict[str, Any]:
    result = _json_object(text)
    expected = {
        "performed_social_behavior",
        "actor_action_target_same_event",
        "target_behavior_performed_not_described",
        "affected_party_or_shared_setting_present",
        "socially_evaluable_without_metadata",
        "social_scope",
        "scene_role",
        "social_response_or_consequence_present",
        "literal_actor",
        "literal_action_or_situated_utterance",
        "literal_affected_party_or_shared_setting",
        "demo_usable",
        "rejection_reason",
        "evidence",
    }
    if set(result) != expected:
        raise ValueError(
            f"schema mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    atomic = [
        "performed_social_behavior",
        "actor_action_target_same_event",
        "target_behavior_performed_not_described",
        "affected_party_or_shared_setting_present",
        "socially_evaluable_without_metadata",
        "social_response_or_consequence_present",
        "demo_usable",
    ]
    for key in atomic:
        if result[key] not in YES_NO_UNCERTAIN:
            raise ValueError(f"invalid {key}: {result[key]!r}")
    enum_repairs: list[str] = []
    enum_specs = {
        "social_scope": (SOCIAL_SCOPES, "uncertain"),
        "scene_role": (SCENE_ROLES, "uncertain"),
        "rejection_reason": (REJECTION_REASONS, "uncertain"),
    }
    for key, (allowed, fallback) in enum_specs.items():
        if result[key] not in allowed:
            result[f"{key}_raw"] = result[key]
            result[key] = fallback
            enum_repairs.append(key)
    strings = [
        "literal_actor",
        "literal_action_or_situated_utterance",
        "literal_affected_party_or_shared_setting",
        "evidence",
    ]
    if any(not isinstance(result[key], str) or not result[key].strip() for key in strings):
        raise ValueError("literal fields and evidence must be non-empty strings")

    strict_positive = (
        all(result[key] == "yes" for key in atomic[:5])
        and result["social_scope"] in QUALIFYING_SCOPES
        and result["scene_role"] in QUALIFYING_ROLES
    )
    if result["demo_usable"] == "yes" and not strict_positive:
        result["demo_usable_raw"] = "yes"
        result["demo_usable"] = "uncertain"
        result["consistency_repair"] = "inconsistent_positive_to_uncertain"
    if result["demo_usable"] == "yes" and result["rejection_reason"] != "none":
        result["demo_usable_raw"] = "yes"
        result["demo_usable"] = "uncertain"
        result["consistency_repair"] = "positive_with_rejection_to_uncertain"
    if enum_repairs:
        if result["demo_usable"] != "uncertain":
            result["demo_usable_raw"] = result["demo_usable"]
        result["demo_usable"] = "uncertain"
        result["enum_repairs"] = enum_repairs
        result["consistency_repair"] = "unknown_enum_to_uncertain_fail_closed"
    return result


def successful_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (row["item_id"], row["model"])
        for row in rows
        if row.get("result") is not None and row.get("error") is None
    }


def base_record(row: dict[str, Any], model: str) -> dict[str, Any]:
    """Preserve only identity fields explicitly present in a frozen manifest."""
    record: dict[str, Any] = {
        "item_id": row["item_id"],
        "pillar": row["pillar"],
        "rubric": "v10a_storyboard",
        "model": model,
        "sheet_sha256": row["sheet_sha256"],
        "frame_count": row["frame_count"],
    }
    for key in ("candidate_id", "uid", "audit_index"):
        if key in row:
            record[key] = row[key]
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
        "max_tokens": 650,
        "messages": [
            {"role": "system", "content": SYSTEM_V10A},
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
                            "Audit this complete dense temporal storyboard. "
                            "Use only the ordered frames and visible text."
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
            parsed = parse_v10a(content)
            return {
                **base_record(row, model),
                "result": parsed,
                "raw_response": content,
                "usage": body.get("usage"),
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode(errors="replace")[:2000]
            except Exception:
                detail = ""
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
        **base_record(row, model),
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8272/v1")
    parser.add_argument("--model", default="glm-4.6v-flash")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--default-pillar",
        default="instructional",
        help="pillar used only when the frozen manifest omits one",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    for row in rows:
        row["_manifest_dir"] = str(args.manifest.parent)
        row.setdefault("pillar", args.default_pillar)
    completed = successful_keys(load_jsonl(args.out)) if args.out.exists() else set()
    pending = [row for row in rows if (row["item_id"], args.model) not in completed]
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
                    **base_record(row, args.model),
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
