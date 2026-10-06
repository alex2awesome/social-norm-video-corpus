#!/usr/bin/env python3
"""Export a frozen witnessed audit cohort for the open-video-VLM runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_rows(root: Path) -> list[dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    adjudication = json.loads(
        (root / "manual_revealed_label_adjudication.json").read_text()
    )
    blind = {
        row["audit_index"]: row
        for row in load_jsonl(root / "manual_blind_witnessed_review.jsonl")
    }
    transcript_artifact = json.loads(
        (root / "all_clip_transcripts_tiny_en.json").read_text()
    )
    transcripts = {row["uid"]: row["segments"] for row in transcript_artifact["records"]}

    strict = adjudication["strict_witnessed_contract"]
    accepted = set(strict["accepted_indices"])
    unresolved = set(strict["source_authenticity_unresolved_indices"])
    rows = []
    for source in manifest["visual_samples"]:
        index = int(source["audit_index"])
        uid = str(source["uid"])
        transcript_key = f"{index}_{uid}"
        if index not in blind:
            raise ValueError(f"missing blind judgment for index {index}")
        if transcript_key not in transcripts:
            raise ValueError(f"missing transcript for {transcript_key}")
        rows.append(
            {
                "item_id": f"witnessed:{uid}:0",
                "pillar": "witnessed",
                "uid": uid,
                "source_clip": f"clips/{index}_{uid}.mp4",
                "proxy_clip": f"proxy_clips_1fps/{index}_{uid}.mp4",
                "norm": source.get("norm"),
                "explanation": (
                    f"Alleged reaction: {source.get('reaction')!r}. "
                    f"Detector context: {source.get('reaction_context')!r}."
                ),
                "aligned_transcript": transcripts[transcript_key],
                "gold_scene_visible": blind[index]["action_visibility"] == "yes",
                "gold_social_scene_visible": (
                    blind[index]["action_visibility"] in {"yes", "needs_motion"}
                ),
                "gold_label_matched_visible": None,
                "gold_usable": index in accepted,
                "gold_source_authenticity_unresolved": index in unresolved,
                "audit_index": index,
            }
        )
    if [row["audit_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("cohort audit indices are not contiguous and ordered")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.root / "vlm_manifest.jsonl"
    rows = build_rows(args.root)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )
    print(json.dumps({"rows": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
