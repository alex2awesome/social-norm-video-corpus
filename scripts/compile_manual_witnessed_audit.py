#!/usr/bin/env python3
"""Compile a complete manual witnessed review TSV into ledger JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from run_openai_witnessed_audit import PROMPT_VERSION, prompt_hash, validate_result


TRI_FIELDS = (
    "is_social_norm", "social_action_visible", "reaction_visible",
    "action_before_reaction", "reaction_targets_action", "reaction_is_normative",
    "behavior_label_supported", "clean_pre_reaction_demo",
)


def list_field(value: str, cast=str):
    return [cast(x) for x in value.split(";") if x]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("review_tsv", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", default="codex-manual-visual-review")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["rubric_version"] != PROMPT_VERSION:
        raise SystemExit("manifest rubric does not match witnessed runner")
    items = {item["item_id"]: item for item in manifest["items"]}
    with args.review_tsv.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != len(items) or {row["item_id"] for row in rows} != set(items):
        raise SystemExit("review TSV must contain every manifest item exactly once")
    records = []
    for row in rows:
        item = items[row["item_id"]]
        result = {key: row[key] for key in TRI_FIELDS}
        result.update({
            "authenticity": row["authenticity"],
            "decision": row["decision"],
            "action_end_sec": float(row["action_end_sec"]) if row["action_end_sec"] else None,
            "reaction_start_sec": float(row["reaction_start_sec"]) if row["reaction_start_sec"] else None,
            "rejection_reasons": list_field(row["rejection_reasons"]),
            "action_evidence_frames": list_field(row["action_evidence_frames"], int),
            "reaction_evidence_frames": list_field(row["reaction_evidence_frames"], int),
            "description": row["manual_description"],
        })
        validate_result(result, item)
        records.append({
            "item_id": row["item_id"],
            "batch_id": manifest["batch_id"],
            "rubric_version": PROMPT_VERSION,
            "model": args.model,
            "prompt_sha256": prompt_hash(),
            "frame_manifest_sha256": item["frame_manifest_sha256"],
            "pass_index": 0,
            **result,
            "raw_response": {
                "review_type": "manual witnessed expansion audit",
                "final_gpt_5_6_sol": False,
                "source_review_tsv": str(args.review_tsv),
            },
            "audited_at": time.time(),
        })
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(json.dumps({"records": len(records), "prompt_sha256": prompt_hash()}))


if __name__ == "__main__":
    main()
