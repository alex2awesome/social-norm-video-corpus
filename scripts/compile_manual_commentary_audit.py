#!/usr/bin/env python3
"""Compile a complete manual commentary review TSV into ledger JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from run_openai_commentary_audit import PROMPT_VERSION, prompt_hash, validate_result


TRI_FIELDS = (
    "is_social_norm", "concrete_behavior", "social_actor_grounded",
    "target_or_shared_context_grounded", "normative_stance_grounded",
    "proposed_norm_supported",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("review_tsv", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", default="codex-manual-text-review")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["rubric_version"] != PROMPT_VERSION:
        raise SystemExit("manifest rubric does not match commentary runner")
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
            "stance_type": row["stance_type"],
            "quote_relation": row["quote_relation"],
            "decision": row["decision"],
            "rejection_reasons": [x for x in row["rejection_reasons"].split(";") if x],
            "normalized_behavior": row["normalized_behavior"],
            "normalized_norm": row["normalized_norm"],
            "behavior_evidence_quote": row["behavior_evidence_quote"],
            "stance_evidence_quote": row["stance_evidence_quote"],
            "description": row["manual_description"],
        })
        validate_result(result, item)
        records.append({
            "item_id": row["item_id"],
            "batch_id": manifest["batch_id"],
            "rubric_version": PROMPT_VERSION,
            "model": args.model,
            "prompt_sha256": prompt_hash(),
            "content_manifest_sha256": item["content_manifest_sha256"],
            "pass_index": 0,
            **result,
            "raw_response": {
                "review_type": "manual commentary expansion audit",
                "final_gpt_5_6_sol": False,
                "source_review_tsv": str(args.review_tsv),
            },
            "audited_at": time.time(),
        })
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(json.dumps({"records": len(records), "prompt_sha256": prompt_hash()}))


if __name__ == "__main__":
    main()
