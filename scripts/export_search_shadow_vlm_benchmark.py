#!/usr/bin/env python3
"""Freeze a search-shadow manual audit as a contact-sheet VLM benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SOCIAL_SCENE_DECISIONS = {
    "accept_after_localization",
    "accept_after_localization_and_label_mask",
    "accept_after_localization_and_relabel",
    "accept_event_windows_only",
    "accept_event_windows_with_medium_tag",
    "accept_graphic_tier",
    "reroute_action_reservoir",
    "reroute_affected_party_response",
    "reroute_correct_demo",
    "reroute_enforcement_or_instructional",
    "reroute_instructional",
}
VISUAL_CONTENT_VALUES = {
    "yes",
    "yes_action_only",
    "yes_correct_action",
    "yes_graphic",
    "yes_offtopic",
    "mixed",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_records(manifest: dict, reviews: list[dict], artifact_root: Path) -> list[dict]:
    by_index = {row["audit_index"]: row for row in reviews}
    if len(by_index) != len(reviews):
        raise ValueError("manual review contains duplicate audit_index values")
    if set(by_index) != {row["ordinal"] for row in manifest["records"]}:
        raise ValueError("manual review and video manifest ordinals differ")

    output = []
    for row in sorted(manifest["records"], key=lambda value: value["ordinal"]):
        review = by_index[row["ordinal"]]
        if row["uid"] != review["uid"] or row["pillar"] != review["intended_pillar"]:
            raise ValueError(f"manual/manifest identity mismatch at ordinal {row['ordinal']}")
        if row["artifact_status"] != "rendered":
            raise ValueError(f"benchmark item is not rendered: {row['uid']}")
        sheet = artifact_root / row["sheet_path"]
        if not sheet.is_file():
            raise ValueError(f"missing contact sheet: {sheet}")
        output.append(
            {
                "item_id": f"search-shadow-v4:{row['ordinal']:02d}:{row['uid']}",
                "ordinal": row["ordinal"],
                "uid": row["uid"],
                "pillar": row["pillar"],
                "query": row["query"],
                "sheet_path": str(sheet),
                "sheet_sha256": row["sheet_sha256"],
                "gold_visual_content_present": review["visual_scene"]
                in VISUAL_CONTENT_VALUES,
                "gold_social_behavior_scene": review["decision"]
                in SOCIAL_SCENE_DECISIONS,
                "gold_target_source_pass": bool(review["strict_target_pass"]),
                "gold_manual_decision": review["decision"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = build_records(
        json.loads(args.video_manifest.read_text()),
        load_jsonl(args.manual_review),
        args.artifact_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"items": len(records), "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
