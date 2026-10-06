#!/usr/bin/env python3
"""Text-only second pass for instructional V9 shadow scoring.

The first-pass VLM record is treated as immutable visual evidence. This script
never sends video to the alignment model and never edits corpus membership.
Append-only output makes interrupted scoring resumable and auditable.
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


SYSTEM = """You are the TEXT-ONLY alignment pass of a two-pass instructional
video audit. You receive:
1. a frozen, label-blind record of the literal event seen in a clip; and
2. fallible metadata proposing a norm and polarity.

The frozen visual record is the complete visual evidence. Never add an actor,
action, target, utterance, object transfer, reaction, or setting that is not in
that record. Do not reinterpret generic footage using the title or explanation.
If the record says no/uncertain event, unlocalized, presentation, montage,
aftermath, generic B-roll, or metadata-needed, the proposed label cannot make it
a visual demo.

Any clearly performed demonstration format may qualify. Explicit social
etiquette (for example greeting, dining, hosting, or culturally situated
interaction conventions) is a valid separate band when the literal behavior is
performed. Exclude motor/game technique, object handling, legal/safety/traffic
rules, technical procedures, and generic values unless the frozen event itself
contains distinct socially evaluable interpersonal/shared-public conduct.

Judge three things separately:
- whether the frozen event is itself a usable social-norm demonstration;
- whether the proposed norm denotes that same literal behavior; and
- whether the proposed polarity matches the event rather than merely describing
  a lesson around it.

Return one JSON object with exactly:
visual_demo_status: exact|usable_after_relabel|no_visual_demo|uncertain
proposed_norm_relation: exact|broader_but_supported|related_not_demonstrated|mismatch|uncertain
proposed_polarity_relation: matches_event|opposes_event|explanation_not_event_polarity|unjudgeable|uncertain
visible_event_polarity: correct|violation|neutral_or_ambiguous|uncertain
social_norm_type: tacit_interpersonal|shared_public|explicit_social_etiquette|formal_rule_or_procedure|motor_game_or_technical|abstract_or_off_topic|uncertain
exact_weak_label_usable: yes|no|uncertain
usable_after_relabel: yes|no|uncertain
normalized_visible_behavior: concise actor-neutral literal behavior or none
normalized_visible_polarity: correct|violation|neutral_or_ambiguous|uncertain|none
mismatch_reason: none|no_localized_visual_event|label_supplied_action|generic_context_only|formal_or_procedural|motor_or_technical|abstract_label|polarity_mismatch|different_visible_event|uncertain
alignment_evidence: one short sentence comparing the proposed label only to the frozen literal action

The fields must agree. exact_weak_label_usable=yes requires a strict frozen
visual event, a qualifying social norm type, proposed_norm_relation=exact, and
no polarity contradiction. usable_after_relabel=yes requires the same strict
visual event, a qualifying social norm type, and a nonempty normalized visible
behavior even when the proposed label is wrong. Otherwise fail closed."""


YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}
QUALIFYING_NORM_TYPES = {
    "tacit_interpersonal",
    "shared_public",
    "explicit_social_etiquette",
}
EVENT_EVIDENCE_SOURCES = {
    "physical_action",
    "situated_dialogue_or_subtitles",
    "static_depicted_action",
}
EVENT_DEMO_KINDS = {
    "interpersonal_conduct",
    "shared_public_conduct",
    "explicit_social_etiquette",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def successful_item_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        row["item_id"]
        for row in rows
        if row.get("result") is not None and row.get("error") is None
    }


def index_visual_records(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("rubric") not in {"v9a", "v9a_storyboard"}:
            continue
        if row.get("result") is None or row.get("error") is not None:
            continue
        item_id = row["item_id"]
        if item_id in result:
            raise ValueError(f"duplicate successful V9A record for {item_id}")
        result[item_id] = row
    return result


def strict_visual_event(visual: dict[str, Any]) -> bool:
    result = visual["result"]
    literal_fields_present = all(
        str(result.get(key, "")).strip().lower() not in {"", "none"}
        for key in (
            "literal_actor",
            "literal_action_or_situated_utterance",
            "literal_affected_party_or_shared_setting",
        )
    )
    start = result.get("event_start_percent")
    end = result.get("event_end_percent")
    return (
        result.get("observable_event") == "yes"
        and isinstance(start, (int, float))
        and isinstance(end, (int, float))
        and 0 <= start <= end <= 100
        and result.get("same_event_actor_action_target") == "yes"
        and result.get("event_temporally_localized") == "yes"
        and result.get("evidence_source") in EVENT_EVIDENCE_SOURCES
        and result.get("scene_role") == "demonstrated_event"
        and result.get("demonstration_kind") in EVENT_DEMO_KINDS
        and literal_fields_present
    )


def parse_json(text: str, visual: dict[str, Any]) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contained no JSON object")
    result = json.loads(text[start : end + 1])
    required = {
        "visual_demo_status",
        "proposed_norm_relation",
        "proposed_polarity_relation",
        "visible_event_polarity",
        "social_norm_type",
        "exact_weak_label_usable",
        "usable_after_relabel",
        "normalized_visible_behavior",
        "normalized_visible_polarity",
        "mismatch_reason",
        "alignment_evidence",
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    if result["proposed_polarity_relation"] == "polarity_mismatch":
        result["proposed_polarity_relation"] = "opposes_event"
        result["proposed_polarity_relation_validation_repair"] = (
            "polarity_mismatch_to_opposes_event_fail_closed"
        )
    if result["proposed_norm_relation"] == "abstract_label":
        result["proposed_norm_relation"] = "mismatch"
        result["proposed_norm_relation_validation_repair"] = (
            "abstract_label_to_mismatch_fail_closed"
        )
    if result["mismatch_reason"] == "broader_but_supported":
        # The relation field already preserves the non-exact judgment. "none"
        # is the only compatible reason token and cannot by itself promote a
        # record through either programmatic gate.
        result["mismatch_reason"] = "none"
        result["mismatch_reason_validation_repair"] = (
            "broader_but_supported_to_none_relation_preserved"
        )
    allowed = {
        "visual_demo_status": {
            "exact",
            "usable_after_relabel",
            "no_visual_demo",
            "uncertain",
        },
        "proposed_norm_relation": {
            "exact",
            "broader_but_supported",
            "related_not_demonstrated",
            "mismatch",
            "uncertain",
        },
        "proposed_polarity_relation": {
            "matches_event",
            "opposes_event",
            "explanation_not_event_polarity",
            "unjudgeable",
            "uncertain",
        },
        "visible_event_polarity": {
            "correct",
            "violation",
            "neutral_or_ambiguous",
            "uncertain",
        },
        "social_norm_type": {
            *QUALIFYING_NORM_TYPES,
            "formal_rule_or_procedure",
            "motor_game_or_technical",
            "abstract_or_off_topic",
            "uncertain",
        },
        "exact_weak_label_usable": YES_NO_UNCERTAIN,
        "usable_after_relabel": YES_NO_UNCERTAIN,
        "normalized_visible_polarity": {
            "correct",
            "violation",
            "neutral_or_ambiguous",
            "uncertain",
            "none",
        },
        "mismatch_reason": {
            "none",
            "no_localized_visual_event",
            "label_supplied_action",
            "generic_context_only",
            "formal_or_procedural",
            "motor_or_technical",
            "abstract_label",
            "polarity_mismatch",
            "different_visible_event",
            "uncertain",
        },
    }
    out_of_schema_fields = []
    for key, values in allowed.items():
        if result[key] not in values:
            if "uncertain" not in values:
                raise ValueError(f"invalid V9B {key}: {result[key]!r}")
            raw_value = result[key]
            result[key] = "uncertain"
            result[f"{key}_validation_repair"] = (
                f"out_of_schema_{raw_value}_to_uncertain_fail_closed"
            )
            out_of_schema_fields.append(key)

    visually_ready = strict_visual_event(visual)
    qualifying_norm = result["social_norm_type"] in QUALIFYING_NORM_TYPES
    behavior_present = (
        str(result["normalized_visible_behavior"]).strip().lower()
        not in {"", "none"}
    )
    no_polarity_contradiction = (
        result["proposed_polarity_relation"] != "opposes_event"
    )
    internally_exact = (
        visually_ready
        and qualifying_norm
        and result["proposed_norm_relation"] == "exact"
        and no_polarity_contradiction
    )
    internally_relabelable = visually_ready and qualifying_norm and behavior_present

    if result["exact_weak_label_usable"] == "yes" and not internally_exact:
        result["exact_weak_label_usable"] = "uncertain"
        result["exact_weak_label_validation_repair"] = (
            "inconsistent_yes_to_uncertain_fail_closed"
        )
    if result["usable_after_relabel"] == "yes" and not internally_relabelable:
        result["usable_after_relabel"] = "uncertain"
        result["usable_after_relabel_validation_repair"] = (
            "inconsistent_yes_to_uncertain_fail_closed"
        )
    if result["visual_demo_status"] == "exact" and not internally_exact:
        result["visual_demo_status"] = "uncertain"
        result["visual_demo_status_validation_repair"] = (
            "inconsistent_exact_to_uncertain_fail_closed"
        )
    if (
        result["visual_demo_status"] == "usable_after_relabel"
        and not internally_relabelable
    ):
        result["visual_demo_status"] = "uncertain"
        result["visual_demo_status_validation_repair"] = (
            "inconsistent_relabel_to_uncertain_fail_closed"
        )
    if out_of_schema_fields:
        # An out-of-schema semantic judgment is a contract failure, not
        # evidence for inclusion. Preserve the repaired values for audit while
        # making every programmatic selection gate non-positive.
        for gate in (
            "exact_weak_label_usable",
            "usable_after_relabel",
            "visual_demo_status",
        ):
            result[gate] = "uncertain"
        result["out_of_schema_validation_repair"] = {
            "fields": sorted(out_of_schema_fields),
            "selection_gates_forced": "uncertain",
        }
    return result


def request_one(
    endpoint: str,
    model: str,
    row: dict[str, Any],
    visual: dict[str, Any],
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    frozen = visual["result"]
    prompt = (
        "Frozen label-blind visual record (the only visual evidence):\n"
        f"{json.dumps(frozen, sort_keys=True, ensure_ascii=False)}\n\n"
        "Fallible metadata to align:\n"
        f"Proposed norm: {row.get('norm') or ''}\n"
        f"Proposed polarity: {row.get('polarity') or ''}\n"
        f"Grounded start quote: {str(row.get('start_quote') or '')[:1000]}\n"
        f"Grounded end quote: {str(row.get('end_quote') or '')[:1000]}\n"
        f"Detector explanation: {str(row.get('explanation') or '')[:1400]}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 600,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
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
            parsed = parse_json(content, visual)
            return {
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": row["pillar"],
                "rubric": "v9b",
                "model": model,
                "visual_model": visual.get("model"),
                "visual_record_sha256": canonical_sha256(visual),
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
        "item_id": row["item_id"],
        "uid": row["uid"],
        "pillar": row["pillar"],
        "rubric": "v9b",
        "model": model,
        "visual_model": visual.get("model"),
        "visual_record_sha256": canonical_sha256(visual),
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8078/v1")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest = load_jsonl(args.manifest)
    visual = index_visual_records(load_jsonl(args.visual_scores))
    completed = (
        successful_item_ids(load_jsonl(args.out)) if args.out.exists() else set()
    )
    pending = [
        row
        for row in manifest
        if row["item_id"] in visual and row["item_id"] not in completed
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
                visual[row["item_id"]],
                args.timeout,
                args.retries,
            ): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            result = future.result()
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{index}/{len(pending)} {row['item_id']} "
                f"{'ok' if result['error'] is None else result['error']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
