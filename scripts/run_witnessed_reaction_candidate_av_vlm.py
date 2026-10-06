#!/usr/bin/env python3
"""Ask an audiovisual VLM for atomic witnessed-reaction judgments (V3)."""

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


SYSTEM_V3 = """You audit one short audiovisual candidate window for a bystander
reaction. Use the video and its audio. Transcript text is imperfect ASR and may
help identify the meaning and timing of speech, but words alone never establish
which visible person spoke or whether that person is a bystander.

Separate these questions:
1. Is a response visibly or audibly grounded in the on-scene event?
2. Is the responder a separate bystander, rather than the actor or directly
   affected target? Authority/host is a separate role. If identity cannot be
   bound from the audiovisual evidence, say offscreen_or_unresolved.
3. Does the response target an earlier/overlapping action?
4. Is the triggering action interpersonal treatment/shared-public conduct, or
   merely an accident, private physical safety, or a codified/legal rule?
5. Is the source clearly staged? Do not infer staging from polished video alone;
   require acting, reveal, reenactment, animation, or production evidence.

Generic surprise, fear, concern after an accident, a target's complaint, an
actor's defense, narration, and unrelated speech are not bystander reactions.
Return exactly one JSON object with these keys:
reaction_grounded: yes|no|uncertain
action_before_or_overlaps_response: yes|no|uncertain
response_targets_action: yes|no|uncertain
responder_role: separate_bystander|organic_audience|authority_or_host|affected_target|violator_or_actor|offscreen_or_unresolved|none
response_content: targeted_objection|correction_or_sanction|protective_intervention|interposition_or_separation|generic_affect|self_defense_or_excuse|description_only|none|uncertain
trigger_kind: interpersonal_treatment|shared_public_conduct|private_physical_safety|accident_or_involuntary|codified_or_legal_only|not_established|uncertain
staging: clearly_staged|no_clear_staging_evidence|uncertain
evidence: one short sentence grounded in the audiovisual window
"""

SYSTEM_V4_SILENT_VIDEO_ASR = """You audit one short silent-video candidate
window plus separately supplied ASR context for a bystander reaction. The video
proxy has no audio track. Never claim to hear a voice, sound, tone, or other
audio evidence. ASR can establish candidate words and approximate timing, but
ASR alone cannot establish which visible person spoke, whether the speaker was
on scene, or whether the speaker was a bystander.

Separate these questions:
1. Is a response grounded by visible behavior and/or the supplied ASR in the
   on-scene event? If grounding comes only from ASR and visible speaker identity
   is unresolved, preserve that uncertainty in responder_role.
2. Is the responder visibly a separate bystander, rather than the actor or
   directly affected target? Authority/host is a separate role. If identity
   cannot be bound from the silent video, say offscreen_or_unresolved.
3. Does the response target an earlier/overlapping action?
4. Is the triggering action interpersonal treatment/shared-public conduct, or
   merely an accident, private physical safety, or a codified/legal rule?
5. Is the source clearly staged? Do not infer staging from polished video alone;
   require acting, reveal, reenactment, animation, or production evidence.

Generic surprise, fear, concern after an accident, a target's complaint, an
actor's defense, narration, and unrelated speech are not bystander reactions.
Return exactly one JSON object with these keys:
reaction_grounded: yes|no|uncertain
action_before_or_overlaps_response: yes|no|uncertain
response_targets_action: yes|no|uncertain
responder_role: separate_bystander|organic_audience|authority_or_host|affected_target|violator_or_actor|offscreen_or_unresolved|none
response_content: targeted_objection|correction_or_sanction|protective_intervention|interposition_or_separation|generic_affect|self_defense_or_excuse|description_only|none|uncertain
trigger_kind: interpersonal_treatment|shared_public_conduct|private_physical_safety|accident_or_involuntary|codified_or_legal_only|not_established|uncertain
staging: clearly_staged|no_clear_staging_evidence|uncertain
evidence: one short sentence naming the visible and/or ASR evidence without
claiming that the silent proxy contains audio
"""

SYSTEM_V5_REAL_AUDIO_ASR = """You audit one short low-frame-rate video window
for a bystander reaction. When the request says original source audio is
present, use that audio together with the visible video and supplied ASR. When
it says audio is absent, never infer sounds or voices. ASR can establish words
and approximate timing, but do not bind a speaker to a visible person unless
the audiovisual evidence supports that identity.

Separate these questions:
1. Is a response visibly and/or audibly grounded in the on-scene event?
2. Is the responder a separate bystander, rather than the actor or directly
   affected target? Authority/host is a separate role. If identity cannot be
   bound from audiovisual evidence, say offscreen_or_unresolved.
3. Does the response target an earlier/overlapping action?
4. Is the triggering action interpersonal treatment/shared-public conduct, or
   merely an accident, private physical safety, or a codified/legal rule?
5. Is the source clearly staged? Do not infer staging from polished video alone;
   require acting, reveal, reenactment, animation, or production evidence.

Generic surprise, fear, concern after an accident, a target's complaint, an
actor's defense, narration, and unrelated speech are not bystander reactions.
Return exactly one JSON object with these keys:
reaction_grounded: yes|no|uncertain
action_before_or_overlaps_response: yes|no|uncertain
response_targets_action: yes|no|uncertain
responder_role: separate_bystander|organic_audience|authority_or_host|affected_target|violator_or_actor|offscreen_or_unresolved|none
response_content: targeted_objection|correction_or_sanction|protective_intervention|interposition_or_separation|generic_affect|self_defense_or_excuse|description_only|none|uncertain
trigger_kind: interpersonal_treatment|shared_public_conduct|private_physical_safety|accident_or_involuntary|codified_or_legal_only|not_established|uncertain
staging: clearly_staged|no_clear_staging_evidence|uncertain
evidence: one short sentence grounded in the video, actual audio when present,
and/or supplied ASR
"""

SYSTEM_V6_ROLE_CAUSAL_BINDING = """You audit one short audiovisual window for
a possible bystander reaction. The central task is causal role binding, not
detecting emotional or normative words. Use the entire window in time order.

First reason privately through this role graph:
- TRIGGER ACTOR: who performs the earlier or overlapping action?
- DIRECT TARGET: who receives that action or is directly addressed by it?
- RESPONSE ACTOR: who produces the candidate response?
- PRIOR PARTICIPATION: was the response actor already a host, interviewer,
  performer, provocateur, companion in the setup, trigger actor, or target?

Label separate_bystander only when audiovisual evidence positively establishes
that the response actor is neither endpoint of the triggering action and was
not participating in the setup. A person does not become a bystander merely
because another person is visible. If speaker identity, trigger actor, target,
or prior participation cannot be resolved, use offscreen_or_unresolved. ASR
establishes approximate words and timing but cannot by itself bind a speaker.

Normative phrases such as "stop," "don't," "that's wrong," or "what are you
doing" do not prove objection. The response must target a concrete visible or
audible earlier/overlapping action. Reject ordinary questions, answers,
self-defense, interview elicitation, playful banter, setup dialogue, and the
violator's own normative demand. Preserve check-welfare/protective reactions
after accidents as reactions, but label the trigger accident_or_involuntary;
social-norm eligibility is a later decision.

Keep staging independent from reaction presence. A staged scene can contain a
real or enacted bystander response, but clearly_staged is not organic witnessed
footage. When the request says original source audio is present, use it with
the video and ASR; when absent, never claim to hear audio.

Return exactly one JSON object with these keys:
reaction_grounded: yes|no|uncertain
action_before_or_overlaps_response: yes|no|uncertain
response_targets_action: yes|no|uncertain
responder_role: separate_bystander|organic_audience|authority_or_host|affected_target|violator_or_actor|offscreen_or_unresolved|none
response_content: targeted_objection|correction_or_sanction|protective_intervention|interposition_or_separation|generic_affect|self_defense_or_excuse|description_only|none|uncertain
trigger_kind: interpersonal_treatment|shared_public_conduct|private_physical_safety|accident_or_involuntary|codified_or_legal_only|not_established|uncertain
staging: clearly_staged|no_clear_staging_evidence|uncertain
evidence: one short sentence naming the trigger actor, target, response actor,
and the concrete action-response relation, or naming what cannot be resolved
"""

SYSTEM_PROMPTS = {
    "v3_audiovisual_wording": SYSTEM_V3,
    "v4_silent_video_asr": SYSTEM_V4_SILENT_VIDEO_ASR,
    "v5_real_audio_asr": SYSTEM_V5_REAL_AUDIO_ASR,
    "v6_role_causal_binding": SYSTEM_V6_ROLE_CAUSAL_BINDING,
}
# Backward-compatible name for imports and frozen V3 behavior.
SYSTEM = SYSTEM_V3

TRINARY = {"yes", "no", "uncertain"}
ROLES = {"separate_bystander", "organic_audience", "authority_or_host", "affected_target", "violator_or_actor", "offscreen_or_unresolved", "none"}
CONTENTS = {"targeted_objection", "correction_or_sanction", "protective_intervention", "interposition_or_separation", "generic_affect", "self_defense_or_excuse", "description_only", "none", "uncertain"}
TRIGGERS = {"interpersonal_treatment", "shared_public_conduct", "private_physical_safety", "accident_or_involuntary", "codified_or_legal_only", "not_established", "uncertain"}
STAGING = {"clearly_staged", "no_clear_staging_evidence", "uncertain"}
EXPECTED = {"reaction_grounded", "action_before_or_overlaps_response", "response_targets_action", "responder_role", "response_content", "trigger_kind", "staging", "evidence"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def successful_candidate_ids(
    rows: list[dict[str, Any]], model: str
) -> set[str]:
    """Collapse append-only attempts so a later success permanently wins."""
    return {
        str(row["candidate_id"])
        for row in rows
        if row.get("model") == model
        and row.get("candidate_id") not in (None, "")
        and row.get("error") in (None, "")
        and isinstance(row.get("result"), dict)
    }


def pending_manifest_rows(
    manifest_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    completed = successful_candidate_ids(output_rows, model)
    return [
        row for row in manifest_rows
        if str(row.get("candidate_id") or "") not in completed
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_result(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        result = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        result = json.loads(stripped[start:end + 1])
    if not isinstance(result, dict) or set(result) != EXPECTED:
        raise ValueError("witnessed reaction AV V3 schema mismatch")
    for key in ("reaction_grounded", "action_before_or_overlaps_response", "response_targets_action"):
        if result[key] not in TRINARY:
            raise ValueError(f"invalid {key}: {result[key]!r}")
    if result["responder_role"] not in ROLES:
        raise ValueError("invalid responder_role")
    if result["response_content"] not in CONTENTS:
        raise ValueError("invalid response_content")
    if result["trigger_kind"] not in TRIGGERS:
        raise ValueError("invalid trigger_kind")
    if result["staging"] not in STAGING:
        raise ValueError("invalid staging")
    if not isinstance(result["evidence"], str) or not result["evidence"].strip():
        raise ValueError("evidence must be non-empty")
    return result


def resolve_media(row: dict[str, Any], media: str) -> tuple[Path, str]:
    field = "candidate_video_path" if media == "video" else "sheet_path"
    hash_field = "candidate_video_sha256" if media == "video" else "sheet_sha256"
    supplied = Path(row[field])
    manifest_dir = Path(row["_manifest_dir"])
    candidates = [supplied]
    if supplied.is_absolute():
        # Some distributed renders retain the producer host's absolute AFS
        # path. The copied artifact lives beside the manifest on the scoring
        # host. Basename fallback is safe only because the content hash below
        # remains mandatory.
        candidates.extend([
            manifest_dir / supplied.name,
            manifest_dir / "media" / supplied.name,
        ])
    else:
        candidates.extend([Path.cwd() / supplied, manifest_dir / supplied])

    def accessible_file(path: Path) -> bool:
        try:
            return path.is_file()
        except OSError:
            return False

    path = next((candidate for candidate in candidates if accessible_file(candidate)), None)
    if path is None:
        raise FileNotFoundError(f"missing {media} artifact: {supplied.name}")
    if sha256(path) != row[hash_field]:
        raise ValueError(f"{media} hash mismatch: {row['candidate_id']}")
    return path.resolve(), hash_field


def retryable_error(exc: Exception) -> bool:
    """Retry transport/server/model-format failures, not deterministic HTTP 4xx."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 409, 425, 429} or exc.code >= 500
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return True
    # A valid response with malformed JSON/schema can vary across deterministic
    # decoding backends and is cheap enough to retry within the bounded budget.
    return isinstance(exc, (ValueError, KeyError, json.JSONDecodeError))


def request_one(endpoint: str, model: str, row: dict[str, Any], timeout: int, retries: int, media: str, prompt_version: str) -> dict[str, Any]:
    media_path, hash_field = resolve_media(row, media)
    before = row.get("context_before") or []
    after = row.get("context_after") or []
    audio_note = ""
    if prompt_version == "v4_silent_video_asr":
        audio_note = "Original source audio present in this media: no. "
    elif prompt_version in {"v5_real_audio_asr", "v6_role_causal_binding"}:
        audio_note = (
            "Original source audio present in this media: "
            f"{'yes' if row.get('audio_present') is True else 'no'}. "
        )
    prompt = (
        f"Candidate timing within this window: {row['candidate_relative_start_sec']:.2f}-"
        f"{row['candidate_relative_end_sec']:.2f}s of {row['window_duration_sec']:.2f}s. "
        f"Candidate ASR: {json.dumps(row.get('candidate_text', ''), ensure_ascii=False)}. "
        f"Prior ASR: {json.dumps(before, ensure_ascii=False)}. "
        f"Following ASR: {json.dumps(after, ensure_ascii=False)}. "
        + audio_note
        + ("The storyboard is silent; do not claim that it proves speaker identity or audibility. " if media == "image" else "")
        + "Audit the atomic questions."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 500,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
            {"role": "user", "content": [
                {"type": f"{media}_url", f"{media}_url": {"url": media_path.as_uri()}},
                {"type": "text", "text": prompt},
            ]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode()
    last_error, last_content = "", None
    for attempt in range(retries + 1):
        request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=encoded, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            last_content = body["choices"][0]["message"].get("content")
            if not last_content:
                raise ValueError("empty response")
            return {"candidate_id": row["candidate_id"], "item_id": row["item_id"], "uid": row["uid"], "model": model, "rubric": f"witnessed_reaction_candidate_{media}_{prompt_version}", "prompt_version": prompt_version, "media_sha256": row[hash_field], "result": parse_result(last_content), "raw_response": last_content, "usage": body.get("usage"), "error": None}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if not retryable_error(exc):
                break
        if attempt < retries:
            time.sleep(2 ** attempt)
    return {"candidate_id": row["candidate_id"], "item_id": row["item_id"], "uid": row["uid"], "model": model, "rubric": f"witnessed_reaction_candidate_{media}_{prompt_version}", "prompt_version": prompt_version, "media_sha256": row[hash_field], "result": None, "raw_response": last_content, "usage": None, "error": last_error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8271/v1")
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--media", choices=("video", "image"), default="video")
    parser.add_argument(
        "--prompt-version",
        choices=tuple(SYSTEM_PROMPTS),
        default="v3_audiovisual_wording",
        help="versioned audit contract; V4 explicitly matches silent proxies",
    )
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    for row in rows:
        row["_manifest_dir"] = str(args.manifest.parent)
    prior_outputs = read_jsonl(args.out) if args.out.exists() else []
    pending = pending_manifest_rows(rows, prior_outputs, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(request_one, args.endpoint, args.model, row, args.timeout, args.retries, args.media, args.prompt_version): row for row in pending}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                row = futures[future]
                result = {"candidate_id": row["candidate_id"], "item_id": row["item_id"], "uid": row["uid"], "model": args.model, "rubric": f"witnessed_reaction_candidate_{args.media}_{args.prompt_version}", "prompt_version": args.prompt_version, "result": None, "raw_response": None, "usage": None, "error": f"{type(exc).__name__}: {exc}"}
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
    # Historical failed attempts remain in the append-only log. Completion is
    # therefore defined by exact manifest coverage with at least one successful
    # attempt, not by requiring every historical row to be successful.
    remaining = pending_manifest_rows(rows, read_jsonl(args.out), args.model)
    return 0 if not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
