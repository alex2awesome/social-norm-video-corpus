#!/usr/bin/env python3
"""Build full-source search packets from manually reviewed commentary labels.

The transcript supplies the actor/action/target label and temporal anchors, but
never certifies visual evidence.  Both behavior and normative-stance quotes
must be found in timestamped transcript segments.  Title and query metadata are
retained only as provenance.  Input labels must already carry an explicit
``text_label_manual_reviewed=true`` flag; this builder does not convert model
predictions into gold labels.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

if __package__:
    from scripts.build_commentary_hierarchical_window_manifest_v2 import (
        read_jsonl,
        write_jsonl,
    )
else:
    from build_commentary_hierarchical_window_manifest_v2 import read_jsonl, write_jsonl


ACCEPTED = {"accept", "accept_after_relabel"}
VISUAL_DEIXIS = re.compile(
    r"\b(?:video|footage|clip)\s+(?:shows?|showed|showing|captures?|captured)\b"
    r"|\b(?:caught|captured|seen)\s+on\s+(?:camera|video)\b"
    r"|\bas\s+you\s+can\s+see\b|\bwatch\s+(?:this|what happens)\b",
    re.IGNORECASE,
)
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def result_for(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    return result if isinstance(result, dict) else row


def validate_text_label(row: dict[str, Any]) -> dict[str, Any]:
    uid = str(row.get("uid") or "")
    if not uid:
        raise ValueError("text label is missing uid")
    if row.get("text_label_manual_reviewed") is not True:
        raise ValueError(f"{uid}: text label lacks completed manual review")
    result = result_for(row)
    if result.get("decision") not in ACCEPTED:
        raise ValueError(f"{uid}: text label is not accepted")
    required_yes = (
        "social_actor_grounded",
        "concrete_behavior",
        "target_or_shared_context_grounded",
        "normative_stance_grounded",
    )
    for field in required_yes:
        if result.get(field) != "yes":
            raise ValueError(f"{uid}: text label lacks {field}")
    for field in (
        "normalized_behavior",
        "behavior_evidence_quote",
        "stance_evidence_quote",
    ):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f"{uid}: text label lacks {field}")
    return result


def transcript_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for segment in payload.get("segments") or []:
        start = segment.get("start")
        end = segment.get("end")
        text = segment.get("text")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not 0 <= float(start) < float(end)
            or not isinstance(text, str)
        ):
            raise ValueError("transcript contains malformed segment")
        output.append({"start": float(start), "end": float(end), "text": text})
    if not output:
        raise ValueError("transcript has no timestamped segments")
    return output


def grounded_quote_span(
    quote: str, segments: list[dict[str, Any]], maximum_segments: int = 4
) -> tuple[float, float]:
    needle = normalized_text(quote)
    if not needle:
        raise ValueError("empty evidence quote")
    for width in range(1, maximum_segments + 1):
        for start_index in range(0, len(segments) - width + 1):
            group = segments[start_index : start_index + width]
            haystack = normalized_text(" ".join(row["text"] for row in group))
            if needle in haystack:
                return float(group[0]["start"]), float(group[-1]["end"])
    raise ValueError(f"evidence quote is not grounded: {quote!r}")


def visual_deixis_anchors(
    segments: list[dict[str, Any]], duration: float, context_sec: float = 8.0
) -> list[dict[str, Any]]:
    output = []
    for segment in segments:
        if not VISUAL_DEIXIS.search(segment["text"]):
            continue
        output.append({
            "kind": "visual_deixis",
            "start_sec": max(0.0, float(segment["start"]) - context_sec),
            "end_sec": min(duration, float(segment["end"]) + context_sec),
            "evidence": segment["text"],
        })
    return output


def index_unique(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output = {str(row.get("uid") or ""): row for row in rows}
    if not all(output) or len(output) != len(rows):
        raise ValueError(f"{name} has missing or duplicate uid")
    return output


def build_packets(
    labels: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    transcript_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    prior_bounds_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not labels:
        raise ValueError("text-label selection is empty")
    label_index = index_unique(labels, "text labels")
    metadata = index_unique(metadata_rows, "metadata")
    transcripts = index_unique(transcript_rows, "transcripts")
    videos = index_unique(video_rows, "video index")
    prior = index_unique(prior_bounds_rows or [], "prior bounds")
    required = set(label_index)
    for name, index in (
        ("metadata", metadata), ("transcripts", transcripts), ("video index", videos)
    ):
        missing = sorted(required - set(index))
        if missing:
            raise ValueError(f"{name} is missing selected uids: {missing[:3]}")

    packets = []
    for uid in sorted(required):
        label_row = label_index[uid]
        result = validate_text_label(label_row)
        meta = metadata[uid]
        transcript = transcript_segments(transcripts[uid])
        video = videos[uid]
        duration = video.get("duration_sec")
        path = str(video.get("source_path") or "")
        if (
            not isinstance(duration, (int, float))
            or not 0 < float(duration) <= 1800
            or not path
            or Path(path).suffix.lower() not in VIDEO_SUFFIXES
        ):
            raise ValueError(f"{uid}: invalid retained video index row")
        duration = float(duration)
        if transcript[-1]["end"] > duration + 10:
            raise ValueError(f"{uid}: transcript substantially exceeds video duration")

        behavior_span = grounded_quote_span(
            result["behavior_evidence_quote"], transcript
        )
        stance_span = grounded_quote_span(
            result["stance_evidence_quote"], transcript
        )
        anchors = [
            {
                "kind": "commentary_statement",
                "start_sec": max(0.0, min(behavior_span[0], stance_span[0]) - 12.0),
                "end_sec": min(duration, max(behavior_span[1], stance_span[1]) + 4.0),
                "evidence": "grounded behavior and normative-stance transcript spans",
            }
        ]
        anchors.extend(visual_deixis_anchors(transcript, duration))
        if uid in prior:
            for bounds in prior[uid].get("bounds") or []:
                start = float(bounds["start_sec"])
                end = float(bounds["end_sec"])
                if not 0 <= start < end <= duration + 1e-6:
                    raise ValueError(f"{uid}: invalid prior VLM bounds")
                anchors.append({
                    "kind": "prior_vlm",
                    "start_sec": start,
                    "end_sec": end,
                    "evidence": str(bounds.get("model") or "prior_vlm"),
                })

        packets.append({
            "uid": uid,
            "item_id": str(label_row.get("item_id") or f"commentary:{uid}:0"),
            "source_path": path,
            "duration_sec": duration,
            "title": str(meta.get("title") or "untitled retained source"),
            "action_label": result["normalized_behavior"].strip(),
            "normalized_norm": str(result.get("normalized_norm") or "").strip(),
            "behavior_evidence_quote": result["behavior_evidence_quote"],
            "stance_evidence_quote": result["stance_evidence_quote"],
            "behavior_quote_span_sec": list(behavior_span),
            "stance_quote_span_sec": list(stance_span),
            "anchors": anchors,
            "query": meta.get("query") or meta.get("found_by_query"),
            "query_source": meta.get("query_source"),
            "text_label_manual_reviewed": True,
            "script_certifies_visual_event": False,
            "policy": "source_search_packet_only_no_visual_or_keep_label",
        })
    return packets


def load_object_rows(directory: Path, uids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for uid in sorted(uids):
        path = directory / f"{uid}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        row = json.loads(path.read_text())
        row["uid"] = uid
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--transcript-dir", type=Path, required=True)
    parser.add_argument("--video-index", type=Path, required=True)
    parser.add_argument("--prior-bounds", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels = read_jsonl(args.labels)
    uids = {str(row.get("uid") or "") for row in labels}
    packets = build_packets(
        labels,
        load_object_rows(args.metadata_dir, uids),
        load_object_rows(args.transcript_dir, uids),
        read_jsonl(args.video_index),
        read_jsonl(args.prior_bounds) if args.prior_bounds else None,
    )
    write_jsonl(args.out, packets)
    print(json.dumps({
        "sources": len(packets),
        "anchors": sum(len(row["anchors"]) for row in packets),
        "script_certifies_visual_event": False,
        "automatic_acceptance": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
