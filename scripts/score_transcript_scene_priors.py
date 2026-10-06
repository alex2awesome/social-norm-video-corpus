#!/usr/bin/env python3
"""Score transcript-only priors for whether a labeled event may be on screen.

These are explicitly priors, never visual proof. Results are append-only and
the source corpus is not modified.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


OFFSCREEN_PATTERNS = {
    "reported_speech": r"\b(said|says|told|reported|claimed|claims|alleged|allegedly)\b",
    "attribution": r"\b(according to|officials say|police say|sources say)\b",
    "retrospective": r"\b(happened|remember|recalled|used to|had been|years ago|last year)\b",
    "news_frame": r"\b(breaking news|our reporter|joining us|in the studio|the report)\b",
}
DEMO_PATTERNS = {
    "explicit_demo": r"\b(role[- ]?play|demonstration|demo|scenario|vignette|act it out)\b",
    "watch_transition": r"\b(watch this|let'?s watch|look at this|here'?s an example|what happens)\b",
    "direct_exchange": r"\b(please|thank you|excuse me|i'?m sorry|can you|would you|stop it)\b",
    "visual_deixis": r"\b(as you can see|on screen|in this footage|the video shows)\b",
}

SYSTEM = """You estimate a transcript-only PRIOR for whether the labeled action is
being directly depicted during the interval. You cannot see the video. Do not
claim visual certainty. Distinguish direct in-scene dialogue or an announced
role-play from a presenter, lecture, news report, or retrospective account that
only describes an event.

Return JSON with exactly:
discourse_mode: direct_interaction|role_play|instructional_narration|retrospective_account|news_report|lecture|unknown
direct_depiction_prior: number from 0 to 1
offscreen_description_prior: number from 0 to 1
evidence: one short phrase from the transcript or "none"
"""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def transcript_window(path: Path, start: float, end: float, padding: float) -> str:
    data = json.loads(path.read_text())
    parts = [
        segment.get("text", "").strip()
        for segment in data.get("segments", [])
        if float(segment.get("end", 0)) >= start - padding
        and float(segment.get("start", 0)) <= end + padding
    ]
    return " ".join(part for part in parts if part)


def aligned_transcript_text(row: dict) -> str:
    segments = row.get("aligned_transcript")
    if not isinstance(segments, list):
        return ""
    return " ".join(
        str(segment.get("text") or "").strip()
        for segment in segments
        if isinstance(segment, dict) and str(segment.get("text") or "").strip()
    )


def regex_features(text: str) -> dict:
    lowered = text.lower()
    result = {}
    for name, pattern in {**OFFSCREEN_PATTERNS, **DEMO_PATTERNS}.items():
        result[name] = len(re.findall(pattern, lowered, flags=re.IGNORECASE))
    result["word_count"] = len(text.split())
    result["question_marks"] = text.count("?")
    result["first_person_count"] = len(
        re.findall(r"\b(i|i'm|i've|me|my|we|we're|our)\b", lowered)
    )
    result["second_person_count"] = len(
        re.findall(r"\b(you|you're|you've|your)\b", lowered)
    )
    return result


def call_llm(endpoint: str, model: str, text: str, label: str, timeout: int) -> dict:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 250,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    f'Proposed label: "{label}"\n'
                    f"Transcript interval:\n{text[:6000]}"
                ),
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    message = body["choices"][0]["message"]
    content = message.get("content")
    if not content:
        raise ValueError(
            "empty response content; message="
            f"{json.dumps(message, sort_keys=True)[:800]}"
        )
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    start, end = content.find("{"), content.rfind("}")
    return json.loads(content[start : end + 1])


def score_one(row: dict, args: argparse.Namespace) -> dict:
    path = args.transcripts / f"{row['uid']}.json"
    aligned_text = aligned_transcript_text(row)
    result = {
        "item_id": row["item_id"],
        "uid": row["uid"],
        "pillar": row["pillar"],
        "gold_scene_visible": row["gold_scene_visible"],
        "gold_social_scene_visible": row["gold_social_scene_visible"],
        "gold_label_matched_visible": row["gold_label_matched_visible"],
        "gold_usable": row["gold_usable"],
        "transcript_path": str(path) if not aligned_text else None,
        "transcript_exists": bool(aligned_text) or path.is_file(),
        "transcript_source": (
            "manifest_aligned" if aligned_text else "uid_transcript_file"
        ),
    }
    if not aligned_text and not path.is_file():
        result.update({"text": "", "regex": {}, "llm": None, "error": "missing_transcript"})
        return result
    if aligned_text:
        text = aligned_text
    else:
        start = float(row.get("start_sec") or row.get("media_start_sec") or 0)
        end = float(row.get("end_sec") or row.get("media_end_sec") or start)
        text = transcript_window(path, start, end, args.padding)
    result.update({"text": text, "regex": regex_features(text), "llm": None, "error": None})
    if args.endpoint and text:
        try:
            result["llm"] = call_llm(
                args.endpoint,
                args.model,
                text,
                row.get("norm") or "unspecified",
                args.timeout,
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--padding", type=float, default=5.0)
    parser.add_argument("--endpoint")
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    completed = (
        {
            row["item_id"]
            for row in load_jsonl(args.out)
            if row.get("error") is None and row.get("regex")
        }
        if args.out.exists()
        else set()
    )
    pending = [row for row in rows if row["item_id"] not in completed]
    if args.limit:
        pending = pending[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_one, row, args): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(f"{index}/{len(futures)} {result['item_id']} {result['error'] or 'ok'}")


if __name__ == "__main__":
    main()
