#!/usr/bin/env python3
"""Compile a fully reviewed instructional TSV into validator-ready JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from run_openai_visual_audit import PROMPT_VERSION, prompt_hash, validate_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("review_tsv", type=Path)
    parser.add_argument("accept_config", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    manifest = json.loads(args.manifest.read_text())
    accept_config = json.loads(args.accept_config.read_text())
    if not isinstance(accept_config, dict):
        raise SystemExit("accept_config must be an object keyed by item_id")
    if manifest.get("rubric_version") != PROMPT_VERSION:
        raise SystemExit(
            f"manifest rubric {manifest.get('rubric_version')!r} != runner {PROMPT_VERSION!r}"
        )
    items = {item["item_id"]: item for item in manifest["items"]}
    with args.review_tsv.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    reviewed_ids = [row["item_id"] for row in rows]
    if len(reviewed_ids) != len(set(reviewed_ids)):
        raise SystemExit("duplicate manual-review item_id")
    if set(reviewed_ids) != set(items):
        missing = sorted(set(items) - set(reviewed_ids))
        extra = sorted(set(reviewed_ids) - set(items))
        raise SystemExit(f"manual coverage mismatch: missing={missing}, extra={extra}")

    records = []
    for row in rows:
        item = items[row["item_id"]]
        disposition = row["shadow_v4_disposition"]
        special = accept_config.get(row["item_id"])
        if disposition.startswith("accept_with_repairs") and special is None:
            raise SystemExit(f"accepted item lacks audited repair details: {row['item_id']}")
        if special is not None and not (
            disposition == "accept" or disposition.startswith("accept_with_repairs")
        ):
            raise SystemExit(f"special acceptance disagrees with TSV: {row['item_id']}")
        decision = special["decision"] if special is not None else (
            "uncertain" if disposition.startswith("uncertain") else "reject"
        )
        accepted = decision in {"accept", "accept_with_repairs"}
        record = {
            "item_id": row["item_id"],
            "batch_id": manifest["batch_id"],
            "rubric_version": PROMPT_VERSION,
            "model": "codex-manual-visual-review",
            "prompt_sha256": prompt_hash(),
            "frame_manifest_sha256": item["frame_manifest_sha256"],
            "pass_index": 0,
            "is_social_norm": row["is_social_norm"],
            "visual_demo_present": row["visual_demo_present"],
            "medium": row["medium"],
            "clip_composition": row["clip_composition"],
            "norm_supported": row["proposed_norm_supported"],
            "polarity_supported": special["polarity_supported"] if accepted else "uncertain",
            "proposed_polarity_matches": (
                "not_applicable"
                if str(item.get("polarity") or "").lower() == "explanation"
                else (
                    special.get("proposed_polarity_matches", "yes")
                    if accepted else "uncertain"
                )
            ),
            "explanation_supports_norm": row["proposed_norm_supported"],
            "dialogue_grounded": special["dialogue_grounded"] if accepted else "no",
            "visual_norm_supported_without_audio": (
                special["visual_norm_supported_without_audio"] if accepted else "no"
            ),
            "narration_leak": row["narration_leak"],
            "label_leak_visible": row["label_leak_visible"],
            "off_topic": row["off_topic"],
            "decision": decision,
            "required_repairs": special["required_repairs"] if accepted else [],
            "trim_start_sec": special.get("trim_start_sec") if accepted else None,
            "trim_end_sec": special.get("trim_end_sec") if accepted else None,
            "normalized_behavior": special["normalized_behavior"] if accepted else "",
            "normalized_norm": special["normalized_norm"] if accepted else "",
            "rejection_reasons": [] if accepted else row["rejection_reasons"].split(";"),
            "evidence_frames": [frame["frame_index"] for frame in item["frames"]],
            "description": row["manual_description"],
            "raw_response": {
                "review_type": "manual instructional explanation audit",
                "final_gpt_5_6_sol": False,
                "source_review_tsv": str(args.review_tsv),
            },
        }
        validate_result(record, item)
        records.append(record)
    with args.out.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "records": len(records),
                "accepted": sum(r["decision"].startswith("accept") for r in records),
                "prompt_sha256": prompt_hash(),
            }
        )
    )


if __name__ == "__main__":
    main()
