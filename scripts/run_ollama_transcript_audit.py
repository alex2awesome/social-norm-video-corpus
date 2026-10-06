#!/usr/bin/env python3
"""Run a semantic transcript gate without claiming visual certainty."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCHEMA = {
    "type": "object",
    "properties": {
        "social_norm_topic_supported": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
        "specific_hypothesis_supported": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
        "demonstration_or_example_language_present": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
        "situated_dialogue_or_action_cues_present": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
        "discussion_or_advice_only": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
        "visual_depiction_determined": {
            "type": "string",
            "enum": ["no_transcript_cannot_determine_pixels"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "candidate_windows": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "evidence_type": {
                        "type": "string",
                        "enum": [
                            "situated_dialogue",
                            "narrated_action_sequence",
                            "explicit_example_cue",
                            "discussion_only",
                            "uncertain",
                        ],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["start", "end", "evidence_type", "reason"],
            },
        },
        "evidence": {"type": "string"},
    },
    "required": [
        "social_norm_topic_supported",
        "specific_hypothesis_supported",
        "demonstration_or_example_language_present",
        "situated_dialogue_or_action_cues_present",
        "discussion_or_advice_only",
        "visual_depiction_determined",
        "confidence",
        "candidate_windows",
        "evidence",
    ],
}

SYSTEM_V1 = """You audit transcripts only to create semantic and temporal priors
for a separate visual audit. Never claim that an action is visible from words.

Operational definitions:
- social_norm_topic_supported=yes only for interpersonal or public behavior
  governed by a social expectation, not technical, medical, or generic topics.
- specific_hypothesis_supported=yes only when the transcript supports the
  proposed behavior itself, not merely a neighboring topic or search wording.
- demonstration_or_example_language_present=yes for a role-play, character
  story, narrated action sequence, direct situated dialogue, or an explicit cue
  that a concrete example/video is being presented.
- An ordinary panel happens to take conversational turns, but is not a
  turn-taking demonstration. A presenter listing advice is discussion/advice
  only unless a concrete behavior example or scenario is also described.
- A narrated illustrated/animated moral story can be a demonstration prior when
  it describes the relevant action sequence.
- candidate windows should tightly cover the strongest scenario/dialogue/action
  cues, not broad topic discussion.

Return structured JSON. visual_depiction_determined must always state that a
transcript cannot determine pixels."""

SYSTEM_V2 = """You audit transcripts only to create semantic and temporal priors
for a separate visual audit. Never claim that an action is visible from words.

Operational definitions:
- social_norm_topic_supported=yes only for interpersonal or public behavior
  governed by a social expectation, not technical, medical, or generic topics.
- specific_hypothesis_supported means the CORE SOCIAL-NORM BEHAVIOR FAMILY, not
  every retrieval-query modifier. Ignore format/packaging words such as
  "animated", "role play", "training", "dashcam", "original", and "full video"
  for this field. Examples: a story depicting classmates mocking a child
  supports the bullying behavior family even without bystander intervention; a
  story contrasting stealing and sharing supports the sharing/prosocial family;
  body-camera examples of officers calming encounters support de-escalation.
  A technology panel using the word "share" does not support sharing behavior,
  and a generic civil-disturbance exercise does not support de-escalation.
- demonstration_or_example_language_present=yes when the words themselves form
  or introduce a concrete behavior example: role-play; character story;
  narrated action sequence; direct situated dialogue; or an explicit cue that
  an example/video is being presented. It does not require the literal words
  "example", "role play", or "demonstration".
- An ordinary panel happens to take conversational turns, but is not a
  turn-taking demonstration. A presenter listing advice is discussion/advice
  only unless a concrete behavior example or scenario is also described.
- A narrated illustrated/animated moral story is a demonstration prior when it
  describes the relevant action sequence.
- discussion_or_advice_only=yes only when there is no concrete situated example
  or action sequence anywhere in the transcript.
- candidate windows should tightly cover the strongest scenario/dialogue/action
  cues, not broad topic discussion.

Return structured JSON. visual_depiction_determined must always state that a
transcript cannot determine pixels."""

SYSTEMS = {
    "transcript_gate_v1": SYSTEM_V1,
    "transcript_gate_v2": SYSTEM_V2,
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def successful_prediction_keys(records: list[dict]) -> set[tuple[str, str, str]]:
    """Return successful semantic judgments; errors remain eligible for retry."""
    return {
        (
            record["item_id"],
            record["model"],
            record["rubric_version"],
        )
        for record in records
        if record.get("result") is not None and record.get("error") is None
    }


def index_transcripts(records: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Index item-local transcripts without conflating clips from one source."""
    by_item: dict[str, dict] = {}
    by_uid: dict[str, dict] = {}
    for row in records:
        item_id = row.get("item_id")
        uid = row.get("uid") or row.get("video_id")
        if item_id:
            if item_id in by_item:
                raise ValueError(f"duplicate transcript item_id: {item_id}")
            by_item[item_id] = row
        elif uid:
            if uid in by_uid:
                raise ValueError(f"duplicate legacy transcript uid: {uid}")
            by_uid[uid] = row
        else:
            raise ValueError("transcript record requires item_id or uid/video_id")
    return by_item, by_uid


def transcript_for_row(
    row: dict,
    by_item: dict[str, dict],
    by_uid: dict[str, dict],
) -> dict | None:
    return by_item.get(row["item_id"]) or by_uid.get(row["uid"])


def transcript_text(record: dict) -> str:
    return "\n".join(
        f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text'].strip()}"
        for segment in record["segments"]
    )


def validate_result(result: dict) -> None:
    missing = set(SCHEMA["required"]) - result.keys()
    if missing:
        raise ValueError(f"response missing keys: {sorted(missing)}")
    if result["visual_depiction_determined"] != "no_transcript_cannot_determine_pixels":
        raise ValueError("transcript response claimed visual certainty")
    confidence = float(result["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError(f"confidence out of range: {confidence}")


def request_one(
    endpoint: str,
    model: str,
    row: dict,
    transcript: dict,
    rubric_version: str,
    timeout: int,
    retries: int,
) -> dict:
    if rubric_version not in SYSTEMS:
        raise ValueError(f"unknown rubric version: {rubric_version}")
    prompt = (
        f"Candidate pillar: {row['pillar']}.\n"
        f"Retrieval hypothesis: {row['query']!r}.\n"
        "Treat that hypothesis as fallible.\n\n"
        f"TRANSCRIPT:\n{transcript_text(transcript)}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEMS[rubric_version]},
            {"role": "user", "content": prompt},
        ],
        "format": SCHEMA,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 11},
        "keep_alive": "30m",
    }
    encoded = json.dumps(payload).encode()
    last_error = ""
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/api/chat",
            data=encoded,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            result = json.loads(body["message"]["content"])
            validate_result(result)
            return {
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": row["pillar"],
                "model": model,
                "rubric_version": rubric_version,
                "gold_social_behavior_scene": row["gold_social_behavior_scene"],
                "gold_target_source_pass": row["gold_target_source_pass"],
                "result": result,
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
        "item_id": row["item_id"],
        "uid": row["uid"],
        "pillar": row["pillar"],
        "model": model,
        "rubric_version": rubric_version,
        "gold_social_behavior_scene": row["gold_social_behavior_scene"],
        "gold_target_source_pass": row["gold_target_source_pass"],
        "result": None,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="ollama.com/library/qwen3-vl:8b-instruct")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--rubric-version",
        choices=tuple(SYSTEMS),
        default="transcript_gate_v2",
    )
    parser.add_argument("--pillar", choices=("instructional", "witnessed", "commentary"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    transcript_records = json.loads(args.transcripts.read_text())["records"]
    transcripts_by_item, transcripts_by_uid = index_transcripts(transcript_records)
    completed = (
        successful_prediction_keys(load_jsonl(args.out))
        if args.out.exists()
        else set()
    )
    pending = [
        row
        for row in rows
        if transcript_for_row(row, transcripts_by_item, transcripts_by_uid)
        and (args.pillar is None or row["pillar"] == args.pillar)
        and (row["item_id"], args.model, args.rubric_version) not in completed
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.out.open("a") as handle,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {
            pool.submit(
                request_one,
                args.endpoint,
                args.model,
                row,
                transcript_for_row(row, transcripts_by_item, transcripts_by_uid),
                args.rubric_version,
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
