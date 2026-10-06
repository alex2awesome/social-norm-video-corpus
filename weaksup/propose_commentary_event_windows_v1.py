#!/usr/bin/env python3
"""Propose candidate event windows inside both-signal commentary sources.

For sources that are high-confidence on BOTH the text occurred-event and the
visual event-presence targets, propose temporal windows where the discussed
event is most likely visible: windows around footage-deixis / occurred-event
language in the transcript, plus windows anchored on each supported
statement.  Overlapping windows merge; at most ``max_windows`` per source.

Proposals only.  Every automatic commentary crop/trim mechanism failed its
exact-artifact audit, so nothing is cut here — these windows feed manual or
VLM review, ordered by anchor kind (deixis beats statement proximity).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from weaksup.materialize_corpus_lf_records_v1 import (
        FOOTAGE_DEIXIS,
        OCCURRED_EVENT,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from weaksup.materialize_corpus_lf_records_v1 import FOOTAGE_DEIXIS, OCCURRED_EVENT

PROPOSER_VERSION = "propose_commentary_event_windows_v1"

DEIXIS_PRE_SEC = 5.0
DEIXIS_POST_SEC = 25.0
STATEMENT_PAD_SEC = 20.0
MAX_WINDOWS_PER_SOURCE = 5


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def both_high_uids(model_dir: Path) -> set[str]:
    def hi(name: str) -> set[str]:
        return {
            row["item_id"].split(":")[1]
            for row in iter_jsonl(model_dir / name)
            if row["shadow_band"] == "high_confidence_candidate"
        }

    return hi("posteriors_commentary_occurred_event_supported.jsonl") & hi(
        "posteriors_commentary_event_present_in_source.jsonl"
    )


def candidate_windows(
    segments: list[dict[str, Any]],
    statements: list[dict[str, Any]],
    *,
    max_windows: int = MAX_WINDOWS_PER_SOURCE,
    deixis_pre_sec: float = DEIXIS_PRE_SEC,
    deixis_post_sec: float = DEIXIS_POST_SEC,
    statement_pad_sec: float = STATEMENT_PAD_SEC,
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for segment in segments:
        start = segment.get("start")
        if not isinstance(start, (int, float)):
            continue
        text = str(segment.get("text") or "")
        if FOOTAGE_DEIXIS.search(text):
            raw.append({"start": float(start) - deixis_pre_sec,
                        "end": float(start) + deixis_post_sec,
                        "anchor": "footage_deixis", "anchor_text": text.strip()[:120]})
        elif OCCURRED_EVENT.search(text):
            raw.append({"start": float(start) - deixis_pre_sec,
                        "end": float(start) + deixis_post_sec,
                        "anchor": "occurred_event_language",
                        "anchor_text": text.strip()[:120]})
    for statement in statements:
        start = statement.get("start")
        if isinstance(start, (int, float)):
            raw.append({"start": float(start) - statement_pad_sec,
                        "end": float(start) + statement_pad_sec,
                        "anchor": "statement_proximity",
                        "anchor_text": str(statement.get("quote") or "")[:120]})
    for window in raw:
        window["start"] = max(0.0, window["start"])
    raw.sort(key=lambda w: (w["start"], w["end"]))
    # Merge overlaps; a merged window keeps its strongest anchor.
    strength = {"footage_deixis": 3, "occurred_event_language": 2, "statement_proximity": 1}
    merged: list[dict[str, Any]] = []
    for window in raw:
        if merged and window["start"] <= merged[-1]["end"]:
            last = merged[-1]
            last["end"] = max(last["end"], window["end"])
            if strength[window["anchor"]] > strength[last["anchor"]]:
                last["anchor"], last["anchor_text"] = window["anchor"], window["anchor_text"]
        else:
            merged.append(dict(window))
    merged.sort(key=lambda w: (-strength[w["anchor"]], w["start"]))
    kept = merged[:max_windows]
    kept.sort(key=lambda w: w["start"])
    for window in kept:
        window["start"] = round(window["start"], 3)
        window["end"] = round(window["end"], 3)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deixis-pre-sec", type=float, default=DEIXIS_PRE_SEC,
                        help="footage usually FOLLOWS 'watch this'; 2026-09-01 "
                             "spot-check found pre-roll mostly narration, so v2 "
                             "runs should pass a small value here")
    parser.add_argument("--deixis-post-sec", type=float, default=DEIXIS_POST_SEC)
    parser.add_argument("--statement-pad-sec", type=float, default=STATEMENT_PAD_SEC)
    args = parser.parse_args()
    out_path = args.out_dir / "commentary_event_window_proposals.jsonl"
    summary_path = args.out_dir / "summary.json"
    for path in (out_path, summary_path):
        if path.exists():
            raise FileExistsError(f"output exists: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    uids = both_high_uids(args.model_dir)
    counts = {"both_high_videos": len(uids), "with_windows": 0,
              "windows": 0, "missing_transcript": 0}
    with out_path.open("x") as handle:
        for uid in sorted(uids):
            record_path = args.root / "data" / "discussion" / f"{uid}.json"
            transcript_path = args.root / "data" / "transcripts" / f"{uid}.json"
            statements = []
            if record_path.is_file():
                statements = json.loads(record_path.read_text()).get("statements") or []
            segments = []
            if transcript_path.is_file():
                segments = json.loads(transcript_path.read_text()).get("segments") or []
            else:
                counts["missing_transcript"] += 1
            windows = candidate_windows(
                segments, statements,
                deixis_pre_sec=args.deixis_pre_sec,
                deixis_post_sec=args.deixis_post_sec,
                statement_pad_sec=args.statement_pad_sec,
            )
            if windows:
                counts["with_windows"] += 1
                counts["windows"] += len(windows)
            handle.write(json.dumps({
                "uid": uid, "proposer_version": PROPOSER_VERSION,
                "source_video": str(
                    args.root / "data" / "discussion_video" / f"{uid}.mp4"
                ),
                "candidate_windows": windows,
                "acceptance_label": None, "media_cut": False,
                "policy": "localization_proposals_only_manual_or_vlm_review_required",
            }, sort_keys=True) + "\n")
    summary = {"proposer_version": PROPOSER_VERSION, **counts,
               "policy": "localization_proposals_only"}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
