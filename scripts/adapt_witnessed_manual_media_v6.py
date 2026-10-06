#!/usr/bin/env python3
"""Join blind, original-audio audit media back to sealed V6 ASR context.

The renderer intentionally omits ASR and model scores.  This adapter restores
only the already-sealed candidate text/context needed by the V6 role/causal
critic, while keeping cohort names and prior V3 predictions out of each row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flatten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for clip in rows:
        for candidate in clip.get("candidates") or []:
            output.append({
                **candidate,
                "item_id": str(clip["item_id"]),
                "uid": str(clip["uid"]),
            })
    ids = [str(row.get("candidate_id") or "") for row in output]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("sealed selection has missing or duplicate candidate ids")
    return output


def adapt(
    selection: list[dict[str, Any]], media_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates = flatten(selection)
    media = {str(row.get("candidate_id") or ""): row for row in media_rows}
    if len(media) != len(media_rows) or set(media) != {
        str(row["candidate_id"]) for row in candidates
    }:
        raise ValueError("manual media must exactly cover sealed candidates")
    output = []
    for candidate in candidates:
        item = media[str(candidate["candidate_id"])]
        if item.get("error") or not item.get("manual_media_path") or not item.get("manual_media_sha256"):
            raise ValueError(f"{candidate['candidate_id']}: manual media failed")
        output.append({
            **candidate,
            "candidate_video_path": item["manual_media_path"],
            "candidate_video_sha256": item["manual_media_sha256"],
            "audio_present": item.get("audio_present") is True,
            "media_source": "blind_original_audio_manual_audit_render",
            "prior_model_prediction_exposed": False,
            "automatic_acceptance": False,
            "corpus_mutation_authorized": False,
        })
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = adapt(read_jsonl(args.selection), read_jsonl(args.media_manifest))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = {
        "kind": "witnessed_v6_original_audio_model_manifest",
        "candidates": len(rows),
        "with_audio": sum(row["audio_present"] for row in rows),
        "prior_model_prediction_exposed": False,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
        "manifest_sha256": sha256(args.out),
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
