#!/usr/bin/env python3
"""Compile a complete manual review of rendered instructional repairs.

The output is suitable for ``visual_audit_ledger.py import-instructional-repairs``.
It never edits source clips or corpus metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path


RUBRIC_VERSION = "instructional_repair_v1"
CONTRACT = {
    "accepted_requires": [
        "rendered transform audited in full",
        "social norm",
        "visible temporal demo",
        "resolved violation/correct/contrast polarity",
        "no visible label leak",
        "no explanatory narration leak",
        "concrete normalized behavior and norm",
    ],
    "fail_closed": ["missing", "uncertain", "invalid", "unrendered transform"],
}


def prompt_hash() -> str:
    value = {"version": RUBRIC_VERSION, "contract": CONTRACT}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("review_tsv", type=Path)
    parser.add_argument("accepted_config", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--repair-type", required=True, choices=(
        "tighten_to_demo", "crop_label_overlay", "strip_explanatory_audio"
    ))
    parser.add_argument("--model", default="codex-manual-visual-review")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    items = {item["item_id"]: item for item in manifest["items"]}
    config = json.loads(args.accepted_config.read_text())
    with args.review_tsv.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if {row["item_id"] for row in rows} != set(items):
        raise SystemExit("review TSV must contain every manifest item exactly once")
    records = []
    for row in rows:
        item = items[row["item_id"]]
        raw_decision = row["decision"]
        if raw_decision not in {"accept_recut", "accept", "reject", "uncertain"}:
            raise SystemExit(f"invalid decision for {row['item_id']}: {raw_decision}")
        spec = config.get(row["item_id"], {})
        repairs = spec.get("required_repairs", [])
        if raw_decision in {"accept_recut", "accept"}:
            decision = "accept_after_relabel" if repairs else "accept"
        else:
            decision = raw_decision
        rejection_reasons = [x for x in row.get("rejection_reasons", "").split(";") if x]
        norm_supported = row.get("norm_supported") or "uncertain"
        if norm_supported not in {"yes", "no", "uncertain"}:
            raise SystemExit(f"invalid norm_supported for {row['item_id']}")
        transform = {
            **(item.get("transform") or {}),
            **(spec.get("transform") or {}),
        }
        if args.repair_type == "tighten_to_demo":
            transform = {
                "start_sec": float(item["candidate_start_sec"]),
                "end_sec": float(item["candidate_end_sec"]),
                "source_duration_sec": float(item["duration"]),
                **transform,
            }
        evidence_frames = spec.get(
            "evidence_frames", [frame["frame_index"] for frame in item["frames"]]
        )
        accepted = decision in {"accept", "accept_after_relabel"}
        record = {
            "item_id": row["item_id"],
            "batch_id": manifest["batch_id"],
            "rubric_version": RUBRIC_VERSION,
            "model": args.model,
            "prompt_sha256": prompt_hash(),
            "source_frame_manifest_sha256": item["source_frame_manifest_sha256"],
            "repaired_frame_manifest_sha256": item["frame_manifest_sha256"],
            "pass_index": 0,
            "repair_type": args.repair_type,
            "transform": transform,
            "is_social_norm": spec.get(
                "is_social_norm", "no" if "not_social_norm" in rejection_reasons else "yes"
            ),
            "visual_demo_present": spec.get(
                "visual_demo_present", "no" if "no_visual_demo" in rejection_reasons else "yes"
            ),
            "norm_supported": norm_supported,
            "polarity_supported": row.get("polarity_supported") or "uncertain",
            "label_leak_visible": spec.get("label_leak_visible", "no"),
            "narration_leak": spec.get("narration_leak", "no"),
            "decision": decision,
            "required_repairs": repairs if accepted else [],
            "normalized_behavior": spec.get("normalized_behavior", "") if accepted else "",
            "normalized_norm": spec.get("normalized_norm", "") if accepted else "",
            "rejection_reasons": rejection_reasons,
            "evidence_frames": evidence_frames,
            "description": row["manual_description"],
            "raw_response": {
                "review_type": "manual rendered instructional repair audit",
                "final_gpt_5_6_sol": False,
                "source_review_tsv": str(args.review_tsv),
                "rendered_output_audited": True,
            },
            "audited_at": time.time(),
        }
        if not accepted and spec:
            raise SystemExit(f"accepted_config unexpectedly contains rejected item {row['item_id']}")
        records.append(record)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(json.dumps({"records": len(records), "prompt_sha256": prompt_hash()}))


if __name__ == "__main__":
    main()
